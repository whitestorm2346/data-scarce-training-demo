from sklearn.model_selection import train_test_split


def split_dataset(
    image_paths,
    labels,
    train_ratio=0.10,
    val_ratio=0.05,
    test_ratio=0.85,
    seed=42
):
    total = train_ratio + val_ratio + test_ratio

    if abs(total - 1.0) > 1e-6:
        raise ValueError("train_ratio + val_ratio + test_ratio must be 1.0")

    train_paths, temp_paths, train_labels, temp_labels = train_test_split(
        image_paths,
        labels,
        train_size=train_ratio,
        random_state=seed,
        stratify=labels
    )

    val_relative_ratio = val_ratio / (val_ratio + test_ratio)

    val_paths, test_paths, val_labels, test_labels = train_test_split(
        temp_paths,
        temp_labels,
        train_size=val_relative_ratio,
        random_state=seed,
        stratify=temp_labels
    )

    return {
        "train": (train_paths, train_labels),
        "val": (val_paths, val_labels),
        "test": (test_paths, test_labels),
    }