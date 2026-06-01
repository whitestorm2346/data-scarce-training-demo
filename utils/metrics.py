import os
import json
import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay


def evaluate_model(y_true, y_pred, class_names, output_dir, prefix):
    os.makedirs(output_dir, exist_ok=True)

    accuracy = accuracy_score(y_true, y_pred)
    report = classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        output_dict=True,
        zero_division=0
    )

    cm = confusion_matrix(y_true, y_pred)

    report_path = os.path.join(output_dir, f"{prefix}_report.json")
    cm_path = os.path.join(output_dir, f"{prefix}_confusion_matrix.png")

    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "accuracy": accuracy,
                "classification_report": report
            },
            f,
            indent=4,
            ensure_ascii=False
        )

    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=class_names
    )

    display.plot(values_format="d")
    plt.title(f"{prefix} Confusion Matrix")
    plt.tight_layout()
    plt.savefig(cm_path)
    plt.close()

    return accuracy, report_path, cm_path