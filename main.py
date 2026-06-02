import argparse

from traditional.digit_svm_model import run_traditional_digit
from traditional.face_svm_model import run_traditional_face
from deep_learning.train_digit_cnn import run_deep_learning_digit
from deep_learning.train_face_cnn import run_deep_learning_face
from rich.console import Console


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--task", "-t", choices=["digit", "face"], required=True)
    parser.add_argument("--method", "-m", choices=["traditional", "dl", "compare"], required=True)
    parser.add_argument("--data_dir", "-d", default="")

    parser.add_argument("--train_ratio", type=float, default=0.10)
    parser.add_argument("--val_ratio", type=float, default=0.05)
    parser.add_argument("--test_ratio", type=float, default=0.85)

    parser.add_argument("--img_size", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=64)

    return parser.parse_args()


def main():
    args = parse_args()
    console = Console()

    if args.task == "digit":
        if args.data_dir == "":
            args.data_dir = "data/dataset"

        if args.img_size == 0:
            args.img_size = 28

        if args.method == "traditional":
            run_traditional_digit(args, console)

        elif args.method == "dl":
            run_deep_learning_digit(args, console)

        elif args.method == "compare":
            run_traditional_digit(args, console)
            run_deep_learning_digit(args, console)
            raise NotImplementedError("Deep learning comparison will be added later.")

    elif args.task == "face":
        if args.data_dir == "":
            args.data_dir = "data/archive"

        if args.img_size == 0:
            args.img_size = 96

        if args.method == "traditional":
            run_traditional_face(args, console)

        elif args.method == "dl":
            run_deep_learning_face(args, console)

        elif args.method == "compare":
            run_traditional_face(args, console)
            run_deep_learning_face(args, console)
            raise NotImplementedError("Deep learning comparison will be added later.")


if __name__ == "__main__":
    main()