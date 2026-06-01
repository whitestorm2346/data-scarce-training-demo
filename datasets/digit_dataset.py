from pathlib import Path
from collections import Counter
import matplotlib.pyplot as plt

import cv2


def load_digit_image_paths(data_dir):
    data_dir = Path(data_dir)

    if not data_dir.exists():
        raise FileNotFoundError(f"Dataset folder not found: {data_dir}")

    image_paths = []
    labels = []

    for class_dir in sorted(data_dir.iterdir()):
        if not class_dir.is_dir():
            continue

        if not class_dir.name.isdigit():
            continue

        label = int(class_dir.name)

        for img_path in class_dir.rglob("*"):
            if img_path.suffix.lower() in [
                ".png",
                ".jpg",
                ".jpeg",
                ".bmp"
            ]:
                image_paths.append(str(img_path))
                labels.append(label)

    if len(image_paths) == 0:
        raise RuntimeError(f"No images found in {data_dir}")
    
    print(Counter(labels))

    return image_paths, labels


def read_digit_image(path, img_size=28):

    image = cv2.imread(
        path,
        cv2.IMREAD_UNCHANGED
    )

    if image is None:
        raise RuntimeError(
            f"Failed to read image: {path}"
        )

    # PNG with Alpha Channel
    if len(image.shape) == 3 and image.shape[2] == 4:

        image = image[:, :, 3]

    # RGB image
    elif len(image.shape) == 3:

        image = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2GRAY
        )

    image = cv2.resize(
        image,
        (img_size, img_size)
    )

    return image