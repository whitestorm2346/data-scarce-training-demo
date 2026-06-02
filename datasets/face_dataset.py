from pathlib import Path
import cv2


def load_face_image_paths(data_dir):
    data_dir = Path(data_dir)

    if not data_dir.exists():
        raise FileNotFoundError(f"Dataset folder not found: {data_dir}")

    image_paths = []
    labels = []

    class_dirs = sorted([
        d for d in data_dir.iterdir()
        if d.is_dir()
    ])

    class_to_idx = {
        class_dir.name: idx
        for idx, class_dir in enumerate(class_dirs)
    }

    for class_dir in class_dirs:
        label = class_to_idx[class_dir.name]

        for img_path in class_dir.rglob("*"):
            if img_path.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp"]:
                image_paths.append(str(img_path))
                labels.append(label)

    if len(image_paths) == 0:
        raise RuntimeError(f"No face images found in {data_dir}")

    idx_to_class = {
        idx: name
        for name, idx in class_to_idx.items()
    }

    return image_paths, labels, class_to_idx, idx_to_class


def read_face_image(path, img_size=96):
    image = cv2.imread(path)

    if image is None:
        raise RuntimeError(f"Failed to read image: {path}")

    image = cv2.resize(image, (img_size, img_size))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    return image