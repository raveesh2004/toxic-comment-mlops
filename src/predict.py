"""Quick CLI inference: python src/predict.py "some comment text" """

import argparse

from transformers import pipeline


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("text", help="Comment to classify")
    parser.add_argument("--model-dir", default="model")
    args = parser.parse_args()

    classifier = pipeline("text-classification", model=args.model_dir)
    result = classifier(args.text)[0]
    print(f"{result['label']}  (confidence {result['score']:.3f})")


if __name__ == "__main__":
    main()
