"""Command-line interface for training and prediction."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .pipeline import load_artifact, predict, save_artifact, train_model


def main() -> None:
    parser = argparse.ArgumentParser(description="Heavy equipment selling-price prediction")
    subparsers = parser.add_subparsers(dest="command", required=True)
    train_parser = subparsers.add_parser("train", help="Train a model from a CSV")
    train_parser.add_argument("train_csv", type=Path)
    train_parser.add_argument("--artifact", type=Path, default=Path("artifacts/model.joblib"))
    train_parser.add_argument("--target", default="TargetValue")
    train_parser.add_argument("--id-column", default="TransactionID")
    predict_parser = subparsers.add_parser("predict", help="Create predictions from a CSV")
    predict_parser.add_argument("input_csv", type=Path)
    predict_parser.add_argument("--artifact", type=Path, default=Path("artifacts/model.joblib"))
    predict_parser.add_argument("--output", type=Path, default=Path("artifacts/predictions.csv"))

    args = parser.parse_args()
    if args.command == "train":
        artifact, metrics = train_model(
            pd.read_csv(args.train_csv), target_column=args.target, id_column=args.id_column
        )
        save_artifact(artifact, args.artifact)
        print(f"Saved model to {args.artifact}")
        print(f"Validation RMSLE: {metrics['validation_rmsle']:.6f}")
        return

    artifact = load_artifact(args.artifact)
    frame = pd.read_csv(args.input_csv)
    predictions = predict(artifact, frame)
    output = pd.DataFrame(
        {artifact.id_column: frame[artifact.id_column], artifact.target_column: predictions}
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output, index=False)
    print(f"Saved {len(output):,} predictions to {args.output}")
