import argparse
import json
import math
from pathlib import Path

import pandas as pd


KEY_COLUMNS = ["head", "tail", "label"]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Attach out-of-fold baseline probabilities to BindingDB router-training rows."
    )
    parser.add_argument("--source-csv", required=True)
    parser.add_argument("--prediction-csv", action="append", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--summary-json", default=None)
    parser.add_argument(
        "--drop-missing-source",
        action="store_true",
        help="Drop rows excluded by the embedding filter instead of failing coverage validation.",
    )
    return parser.parse_args()


def normalize_ids(frame):
    frame = frame.copy()
    frame["head"] = frame["head"].astype(str).str.replace("Drug:", "", regex=False)
    frame["tail"] = frame["tail"].astype(str).str.replace("Protein:", "", regex=False)
    frame["label"] = frame["label"].astype(int)
    return frame


def combine_prediction_frames(frames):
    combined = []
    for teacher_fold, frame in enumerate(frames):
        frame = normalize_ids(frame)
        probability_column = "prob" if "prob" in frame.columns else "base_prob"
        if probability_column not in frame.columns:
            raise ValueError(f"Prediction frame {teacher_fold} lacks prob/base_prob")
        keep = frame[KEY_COLUMNS + [probability_column]].copy()
        keep = keep.rename(columns={probability_column: "oof_base_prob"})
        keep["teacher_fold"] = teacher_fold
        combined.append(keep)
    predictions = pd.concat(combined, ignore_index=True)
    duplicated = predictions.duplicated(KEY_COLUMNS, keep=False)
    if duplicated.any():
        examples = predictions.loc[duplicated, KEY_COLUMNS].head(10).to_dict("records")
        raise ValueError(f"Rows received multiple OOF predictions: {examples}")
    return predictions


def load_predictions(paths):
    frames = []
    for path in paths:
        frames.append(pd.read_csv(path))
    return combine_prediction_frames(frames)


def main():
    args = parse_args()
    source = pd.read_csv(args.source_csv, index_col=0)
    source = normalize_ids(source)
    if source.duplicated(KEY_COLUMNS).any():
        raise ValueError("Source contains duplicate head/tail/label rows; deduplicate before cross-fit export")

    predictions = load_predictions(args.prediction_csv)
    merged = source.merge(predictions, on=KEY_COLUMNS, how="left", validate="one_to_one")
    missing = int(merged["oof_base_prob"].isna().sum())
    if missing:
        if not args.drop_missing_source:
            raise ValueError(f"Missing OOF predictions for {missing}/{len(merged)} source rows")
        merged = merged[merged["oof_base_prob"].notna()].reset_index(drop=True)
    if not merged["oof_base_prob"].between(0.0, 1.0).all():
        raise ValueError("OOF probabilities must lie in [0, 1]")

    epsilon = 1e-6
    clipped = merged["oof_base_prob"].clip(epsilon, 1.0 - epsilon)
    merged["oof_base_logit"] = clipped.map(lambda value: math.log(value / (1.0 - value)))

    output_path = Path(args.output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(output_path)

    summary = {
        "rows": int(len(merged)),
        "positive_rows": int(merged["label"].sum()),
        "teacher_folds": int(merged["teacher_fold"].nunique()),
        "dropped_source_rows": missing,
        "mean_oof_base_prob": float(merged["oof_base_prob"].mean()),
        "output_csv": str(output_path.resolve()),
    }
    summary_path = Path(args.summary_json) if args.summary_json else output_path.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="ascii")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
