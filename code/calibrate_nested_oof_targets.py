import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn import metrics
from sklearn.linear_model import LogisticRegression


PROTOCOL = "leave-one-teacher-block-out-platt-v1"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Cross-fit calibration for nested dual-blind OOF teacher margins."
    )
    parser.add_argument("--input-csv", required=True)
    parser.add_argument("--input-summary", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--summary-json", required=True)
    return parser.parse_args()


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_single_teacher(value):
    teachers = sorted({int(part) for part in str(value).split(",") if part.strip()})
    if len(teachers) != 1:
        raise ValueError(
            f"Nested calibration requires exactly one teacher per row, received {value!r}"
        )
    return teachers[0]


def fit_platt(logits, labels):
    logits = np.asarray(logits, dtype=np.float64).reshape(-1, 1)
    labels = np.asarray(labels, dtype=np.int64)
    if np.unique(labels).size != 2:
        raise ValueError("Platt calibration requires both classes")
    model = LogisticRegression(C=1e6, solver="lbfgs", max_iter=2000)
    model.fit(logits, labels)
    return float(model.coef_[0, 0]), float(model.intercept_[0])


def apply_platt(logits, scale, bias):
    return expit(float(scale) * np.asarray(logits, dtype=np.float64) + float(bias))


def crossfit_platt(logits, labels, teacher_ids):
    logits = np.asarray(logits, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    teacher_ids = np.asarray(teacher_ids, dtype=np.int64)
    calibrated = np.full(logits.shape, np.nan, dtype=np.float64)
    parameters = {}
    unique_teachers = sorted(np.unique(teacher_ids).tolist())
    if len(unique_teachers) < 2:
        raise ValueError("Cross-fitted calibration requires at least two teacher blocks")

    for heldout in unique_teachers:
        fit_mask = teacher_ids != heldout
        heldout_mask = teacher_ids == heldout
        scale, bias = fit_platt(logits[fit_mask], labels[fit_mask])
        calibrated[heldout_mask] = apply_platt(logits[heldout_mask], scale, bias)
        parameters[str(heldout)] = {
            "scale": scale,
            "bias": bias,
            "fit_rows": int(fit_mask.sum()),
            "heldout_rows": int(heldout_mask.sum()),
        }

    if not np.isfinite(calibrated).all():
        raise AssertionError("Cross-fitted calibration left non-finite predictions")
    return calibrated, parameters


def expected_calibration_error(labels, probabilities, bins=15):
    labels = np.asarray(labels, dtype=np.int64)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = 0.0
    for index, (lower, upper) in enumerate(zip(edges[:-1], edges[1:])):
        if index == bins - 1:
            mask = (probabilities >= lower) & (probabilities <= upper)
        else:
            mask = (probabilities >= lower) & (probabilities < upper)
        if mask.any():
            total += float(mask.mean()) * abs(
                float(labels[mask].mean()) - float(probabilities[mask].mean())
            )
    return total


def metric_summary(labels, ranking_scores, probabilities):
    precision, recall, _ = metrics.precision_recall_curve(labels, ranking_scores)
    safe = np.clip(probabilities, 1e-12, 1.0 - 1e-12)
    return {
        "auc": float(metrics.roc_auc_score(labels, ranking_scores)),
        "aupr": float(metrics.auc(recall, precision)),
        "average_precision": float(metrics.average_precision_score(labels, ranking_scores)),
        "brier": float(metrics.brier_score_loss(labels, probabilities)),
        "nll": float(metrics.log_loss(labels, safe)),
        "ece_15": expected_calibration_error(labels, probabilities, bins=15),
    }


def route_target_summary(labels, probabilities, gamma=2.0):
    labels = np.asarray(labels, dtype=np.float64)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    rescue = labels * np.power(1.0 - probabilities, gamma)
    suppress = (1.0 - labels) * np.power(probabilities, gamma)
    abstain = np.clip(1.0 - rescue - suppress, 0.0, 1.0)
    routes = np.stack([rescue, suppress, abstain], axis=1)
    hard = routes.argmax(axis=1)
    return {
        "gamma": float(gamma),
        "mean_rescue_mass": float(rescue.mean()),
        "mean_suppress_mass": float(suppress.mean()),
        "mean_abstain_mass": float(abstain.mean()),
        "hard_rescue_rows": int((hard == 0).sum()),
        "hard_suppress_rows": int((hard == 1).sum()),
        "hard_abstain_rows": int((hard == 2).sum()),
    }


def calibrate_frame(frame):
    required = {
        "head",
        "tail",
        "label",
        "oof_base_prob",
        "oof_base_logit",
        "oof_teacher_folds",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"OOF CSV is missing columns: {missing}")

    output = frame.copy()
    labels = output["label"].astype(int).to_numpy()
    raw_logits = output["oof_base_logit"].astype(float).to_numpy()
    teacher_ids = output["oof_teacher_folds"].map(parse_single_teacher).to_numpy()
    if not np.isfinite(raw_logits).all():
        raise ValueError("OOF logits contain non-finite values")

    crossfit_probabilities, fold_parameters = crossfit_platt(
        raw_logits,
        labels,
        teacher_ids,
    )
    global_scale, global_bias = fit_platt(raw_logits, labels)
    global_probabilities = apply_platt(raw_logits, global_scale, global_bias)

    output["oof_base_prob_raw"] = output["oof_base_prob"].astype(float)
    output["oof_base_logit_raw"] = raw_logits
    output["oof_base_prob"] = crossfit_probabilities
    safe_crossfit = np.clip(crossfit_probabilities, 1e-12, 1.0 - 1e-12)
    output["oof_base_logit"] = np.log(safe_crossfit / (1.0 - safe_crossfit))
    output["oof_calibration_protocol"] = PROTOCOL
    output["oof_calibration_heldout_teacher"] = teacher_ids

    summary = {
        "protocol": PROTOCOL,
        "rows": int(len(output)),
        "teacher_count": int(np.unique(teacher_ids).size),
        "raw_metrics": metric_summary(labels, raw_logits, expit(raw_logits)),
        "crossfit_metrics": metric_summary(
            labels,
            output["oof_base_logit"].to_numpy(),
            crossfit_probabilities,
        ),
        "global_calibrated_metrics": metric_summary(
            labels,
            global_scale * raw_logits + global_bias,
            global_probabilities,
        ),
        "raw_route_targets": route_target_summary(labels, expit(raw_logits)),
        "crossfit_route_targets": route_target_summary(labels, crossfit_probabilities),
        "global_calibrated_route_targets": route_target_summary(labels, global_probabilities),
        "heldout_parameters": fold_parameters,
        "global_calibrator": {
            "scale": global_scale,
            "bias": global_bias,
            "fit_rows": int(len(output)),
        },
        "outer_val_accessed": False,
        "outer_test_accessed": False,
    }
    return output, summary


def main():
    args = parse_args()
    input_path = Path(args.input_csv)
    input_summary_path = Path(args.input_summary)
    source_summary = json.loads(input_summary_path.read_text(encoding="ascii"))
    if source_summary.get("outer_val_accessed") or source_summary.get("outer_test_accessed"):
        raise ValueError("Input OOF summary reports outer validation/test access")

    frame = pd.read_csv(input_path, index_col=0)
    calibrated, summary = calibrate_frame(frame)
    summary.update(
        {
            "input_csv": str(input_path.resolve()),
            "input_sha256": sha256_file(input_path),
            "source_protocol": source_summary.get("protocol"),
        }
    )

    output_path = Path(args.output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    calibrated.to_csv(output_path)
    summary["output_csv"] = str(output_path.resolve())
    summary["output_sha256"] = sha256_file(output_path)

    summary_path = Path(args.summary_json)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="ascii")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
