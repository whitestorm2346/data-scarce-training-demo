import os
import joblib

from collections import Counter

from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from rich.console import Console

from datasets.face_dataset import load_face_image_paths
from traditional.features import build_face_hog_features
from utils.split_dataset import split_dataset
from utils.metrics import evaluate_model
from utils.seed import set_seed


def run_traditional_face(args, console):
    set_seed(args.seed)

    console.rule("[bold cyan]Face Recognition: Traditional ML[/bold cyan]")

    image_paths, labels, class_to_idx, idx_to_class = load_face_image_paths(
        args.data_dir
    )

    console.print(f"Class count: {len(class_to_idx)}")
    console.print(f"Total samples: {len(image_paths)}")
    console.print(f"Label distribution: {Counter(labels)}")

    splits = split_dataset(
        image_paths=image_paths,
        labels=labels,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed
    )

    train_paths, train_labels = splits["train"]
    val_paths, val_labels = splits["val"]
    test_paths, test_labels = splits["test"]

    console.print(f"Train samples: {len(train_paths)}")
    console.print(f"Val samples:   {len(val_paths)}")
    console.print(f"Test samples:  {len(test_paths)}")

    console.print("\n[bold yellow]Extracting Face HOG Features[/bold yellow]")

    X_train = build_face_hog_features(
        train_paths,
        img_size=args.img_size,
        desc="Train Face Features",
        cache_file="outputs/features/face/train_hog.npy"
    )

    X_val = build_face_hog_features(
        val_paths,
        img_size=args.img_size,
        desc="Validation Face Features",
        cache_file="outputs/features/face/val_hog.npy"
    )

    X_test = build_face_hog_features(
        test_paths,
        img_size=args.img_size,
        desc="Test Face Features",
        cache_file="outputs/features/face/test_hog.npy"
    )

    console.print(f"Feature shape: {X_train.shape}")

    console.print("\n[bold cyan]Training Face Linear SVM[/bold cyan]")

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("svm", LinearSVC(
            C=1.0,
            max_iter=10000,
            random_state=args.seed
        ))
    ])

    model.fit(X_train, train_labels)

    val_pred = model.predict(X_val)
    test_pred = model.predict(X_test)

    class_names = [
        idx_to_class[i]
        for i in range(len(idx_to_class))
    ]

    output_dir = "outputs/reports"
    model_dir = "outputs/models"

    os.makedirs(model_dir, exist_ok=True)

    val_acc, val_report, val_cm = evaluate_model(
        y_true=val_labels,
        y_pred=val_pred,
        class_names=class_names,
        output_dir=output_dir,
        prefix="face_traditional_val"
    )

    test_acc, test_report, test_cm = evaluate_model(
        y_true=test_labels,
        y_pred=test_pred,
        class_names=class_names,
        output_dir=output_dir,
        prefix="face_traditional_test"
    )

    model_path = os.path.join(model_dir, "face_hog_svm.pkl")
    joblib.dump(model, model_path)

    console.rule("[bold green]Result[/bold green]")
    console.print(f"Validation Accuracy: {val_acc:.4f}")
    console.print(f"Test Accuracy:       {test_acc:.4f}")
    console.print(f"Model saved to:      {model_path}")
    console.print(f"Val report:          {val_report}")
    console.print(f"Test report:         {test_report}")
    console.print(f"Val confusion:       {val_cm}")
    console.print(f"Test confusion:      {test_cm}")