import os
import torch
import torch.nn as nn

from torch.utils.data import Dataset, DataLoader
from rich.console import Console
from rich.progress import track

from datasets.face_dataset import load_face_image_paths, read_face_image
from utils.split_dataset import split_dataset
from utils.metrics import evaluate_model
from utils.seed import set_seed
from deep_learning.face_cnn_model import FaceCNN


class FaceImageDataset(Dataset):
    def __init__(self, image_paths, labels, img_size=96):
        self.image_paths = image_paths
        self.labels = labels
        self.img_size = img_size

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = read_face_image(
            self.image_paths[idx],
            img_size=self.img_size
        )

        image = image.astype("float32") / 255.0
        image = torch.tensor(image).unsqueeze(0)

        label = torch.tensor(
            self.labels[idx],
            dtype=torch.long
        )

        return image, label


def train_one_epoch(model, dataloader, criterion, optimizer, device, epoch):
    model.train()

    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in track(
        dataloader,
        description=f"Epoch {epoch} Training"
    ):
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)

        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    avg_loss = total_loss / total
    accuracy = correct / total

    return avg_loss, accuracy


def evaluate_cnn(model, dataloader, device, description="Evaluating"):
    model.eval()

    y_true = []
    y_pred = []

    with torch.no_grad():
        for images, labels in track(
            dataloader,
            description=description
        ):
            images = images.to(device)

            outputs = model(images)
            preds = outputs.argmax(dim=1).cpu().numpy()

            y_pred.extend(preds.tolist())
            y_true.extend(labels.numpy().tolist())

    return y_true, y_pred


def run_deep_learning_face(args, console):
    set_seed(args.seed)

    console.rule("[bold cyan]Face Recognition: Deep Learning[/bold cyan]")

    image_paths, labels, class_to_idx, idx_to_class = load_face_image_paths(
        args.data_dir
    )

    num_classes = len(class_to_idx)

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

    console.print(f"Class count:   {num_classes}")
    console.print(f"Total samples: {len(image_paths)}")
    console.print(f"Train samples: {len(train_paths)}")
    console.print(f"Val samples:   {len(val_paths)}")
    console.print(f"Test samples:  {len(test_paths)}")

    train_dataset = FaceImageDataset(
        train_paths,
        train_labels,
        img_size=args.img_size
    )

    val_dataset = FaceImageDataset(
        val_paths,
        val_labels,
        img_size=args.img_size
    )

    test_dataset = FaceImageDataset(
        test_paths,
        test_labels,
        img_size=args.img_size
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False
    )

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    console.print(f"\nUsing device: [bold green]{device}[/bold green]")

    model = FaceCNN(
        num_classes=num_classes
    ).to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.001,
        weight_decay=1e-4
    )

    best_val_acc = 0.0
    best_model_path = "outputs/models/face_cnn_best.pth"

    os.makedirs("outputs/models", exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model=model,
            dataloader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            epoch=epoch
        )

        val_true, val_pred = evaluate_cnn(
            model=model,
            dataloader=val_loader,
            device=device,
            description=f"Epoch {epoch} Validation"
        )

        val_correct = sum(
            int(t == p) for t, p in zip(val_true, val_pred)
        )
        val_acc = val_correct / len(val_true)

        console.print(
            f"Epoch {epoch:02d} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Train Acc: {train_acc:.4f} | "
            f"Val Acc: {val_acc:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc

            torch.save(
                model.state_dict(),
                best_model_path
            )

            console.print(
                f"[green]Best model saved: {best_model_path}[/green]"
            )

    console.rule("[bold green]Final Evaluation[/bold green]")

    model.load_state_dict(
        torch.load(best_model_path, map_location=device)
    )

    # 建議人臉 confusion matrix 用數字 label，圖會比較乾淨
    class_names = [
        str(i)
        for i in range(num_classes)
    ]

    val_true, val_pred = evaluate_cnn(
        model=model,
        dataloader=val_loader,
        device=device,
        description="Final Validation"
    )

    test_true, test_pred = evaluate_cnn(
        model=model,
        dataloader=test_loader,
        device=device,
        description="Final Test"
    )

    val_acc, val_report, val_cm = evaluate_model(
        y_true=val_true,
        y_pred=val_pred,
        class_names=class_names,
        output_dir="outputs/reports",
        prefix=f"face_dl_val_{args.epochs}epochs"
    )

    test_acc, test_report, test_cm = evaluate_model(
        y_true=test_true,
        y_pred=test_pred,
        class_names=class_names,
        output_dir="outputs/reports",
        prefix=f"face_dl_test_{args.epochs}epochs"
    )

    console.print(f"Best Validation Accuracy: {best_val_acc:.4f}")
    console.print(f"Final Validation Accuracy: {val_acc:.4f}")
    console.print(f"Final Test Accuracy:       {test_acc:.4f}")
    console.print(f"Model saved to:            {best_model_path}")
    console.print(f"Val report:                {val_report}")
    console.print(f"Test report:               {test_report}")
    console.print(f"Val confusion:             {val_cm}")
    console.print(f"Test confusion:            {test_cm}")