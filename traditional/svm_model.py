import os
import joblib

from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from datasets.digit_dataset import load_digit_image_paths
from traditional.features import build_hog_features
from utils.split_dataset import split_dataset
from utils.metrics import evaluate_model
from utils.seed import set_seed


def run_traditional_digit(args, console):
    set_seed(args.seed)

    print("========== Digit Recognition: Traditional ML ==========")

    image_paths, labels = load_digit_image_paths(args.data_dir)

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
    
    print(train_labels[:20])

    print(f"Total samples: {len(image_paths)}")
    print(f"Train samples: {len(train_paths)}")
    print(f"Val samples:   {len(val_paths)}")
    print(f"Test samples:  {len(test_paths)}")

    console.print(
        "\n[bold yellow]Extracting HOG Features[/bold yellow]"
    )

    X_train = build_hog_features(
        train_paths,
        img_size=args.img_size,
        desc="Train Features",
        cache_file="outputs/features/train_hog.npy"
    )

    X_val = build_hog_features(
        val_paths,
        img_size=args.img_size,
        desc="Validation Features",
        cache_file="outputs/features/val_hog.npy"
    )

    X_test = build_hog_features(
        test_paths,
        img_size=args.img_size,
        desc="Test Features",
        cache_file="outputs/features/test_hog.npy"
    )

    print(X_train.shape)
    print(X_train[0][:20])
    print(X_train.min())
    print(X_train.max())

    console.print(
        "\n[bold cyan]Training Linear SVM[/bold cyan]"
    )

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("svm", LinearSVC(
            C=1.0,
            max_iter=5000,
            random_state=args.seed
        ))
    ])

    model.fit(X_train, train_labels)

    print(model.classes_)

    val_pred = model.predict(X_val)
    test_pred = model.predict(X_test)

    class_names = [str(i) for i in range(10)]

    output_dir = "outputs/reports"
    model_dir = "outputs/models"

    os.makedirs(model_dir, exist_ok=True)

    val_acc, val_report, val_cm = evaluate_model(
        y_true=val_labels,
        y_pred=val_pred,
        class_names=class_names,
        output_dir=output_dir,
        prefix="digit_traditional_val"
    )

    test_acc, test_report, test_cm = evaluate_model(
        y_true=test_labels,
        y_pred=test_pred,
        class_names=class_names,
        output_dir=output_dir,
        prefix="digit_traditional_test"
    )

    model_path = os.path.join(model_dir, "digit_hog_svm.pkl")
    joblib.dump(model, model_path)

    console.rule("[bold green]Result[/bold green]")
    
    print(f"Validation Accuracy: {val_acc:.4f}")
    print(f"Test Accuracy:       {test_acc:.4f}")
    print(f"Model saved to:      {model_path}")
    print(f"Val report:          {val_report}")
    print(f"Test report:         {test_report}")
    print(f"Val confusion:       {val_cm}")
    print(f"Test confusion:      {test_cm}")