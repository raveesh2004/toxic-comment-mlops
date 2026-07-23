"""Fine-tune DistilBERT for toxic/offensive comment detection.

Every run is tracked in MLflow: hyperparameters, per-epoch validation metrics,
final test metrics, and the trained model as an artifact.

Usage:
    python src/train.py                       # full run (~12k examples)
    python src/train.py --max-train-samples 2000 --epochs 1   # quick CPU run
"""

import argparse

import mlflow
import numpy as np
from datasets import load_dataset
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

ID2LABEL = {0: "not_offensive", 1: "offensive"}
LABEL2ID = {v: k for k, v in ID2LABEL.items()}


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds),
        "precision": precision_score(labels, preds),
        "recall": recall_score(labels, preds),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", default="distilbert-base-uncased")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--output-dir", default="model")
    args = parser.parse_args()

    # tweet_eval/offensive: binary offensive-language detection, ~12k train
    dataset = load_dataset("tweet_eval", "offensive")
    if args.max_train_samples:
        dataset["train"] = dataset["train"].select(range(args.max_train_samples))

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)

    def tokenize(batch):
        return tokenizer(batch["text"], truncation=True, max_length=args.max_length)

    tokenized = dataset.map(tokenize, batched=True)

    model = AutoModelForSequenceClassification.from_pretrained(
        args.base_model, num_labels=2, id2label=ID2LABEL, label2id=LABEL2ID
    )

    training_args = TrainingArguments(
        output_dir="checkpoints",
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size * 2,
        learning_rate=args.learning_rate,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        logging_steps=50,
        report_to="none",  # MLflow is logged manually below
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=compute_metrics,
    )

    mlflow.set_experiment("toxic-comment-classifier")
    with mlflow.start_run():
        mlflow.log_params(
            {
                "base_model": args.base_model,
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "learning_rate": args.learning_rate,
                "max_length": args.max_length,
                "train_samples": len(tokenized["train"]),
            }
        )

        trainer.train()

        val_metrics = trainer.evaluate(tokenized["validation"])
        test_metrics = trainer.evaluate(tokenized["test"], metric_key_prefix="test")
        mlflow.log_metrics(
            {k: v for k, v in {**val_metrics, **test_metrics}.items() if isinstance(v, float)}
        )

        trainer.save_model(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)
        mlflow.log_artifacts(args.output_dir, artifact_path="model")

        print(f"\nTest metrics: {test_metrics}")
        print(f"Model saved to {args.output_dir}/ and logged to MLflow.")


if __name__ == "__main__":
    main()
