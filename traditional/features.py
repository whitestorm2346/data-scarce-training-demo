import os
import numpy as np
from skimage.feature import hog
from rich.progress import track

from datasets.digit_dataset import read_digit_image


def extract_hog_feature(image):
    feature = hog(
        image,
        orientations=9,
        pixels_per_cell=(7, 7),
        cells_per_block=(2, 2),
        block_norm="L2-Hys"
    )

    return feature


def build_hog_features(
    image_paths,
    img_size=28,
    desc="Extracting HOG",
    cache_file=None
):
    if cache_file is not None and os.path.exists(cache_file):
        print(f"Loading cache: {cache_file}")

        return np.load(cache_file)

    features = []

    for path in track(
        image_paths,
        description=desc
    ):
        image = read_digit_image(
            path,
            img_size=img_size
        )

        feature = extract_hog_feature(image)

        features.append(feature)

    features = np.array(features, dtype=np.float32)

    if cache_file is not None:

        os.makedirs(
            os.path.dirname(cache_file),
            exist_ok=True
        )

        np.save(cache_file, features)

    return features