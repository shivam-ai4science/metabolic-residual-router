#!/usr/bin/env bash
set -Eeuo pipefail

OUTER_SET_ID="${OUTER_SET_ID:-0}"
START_TEACHER="${START_TEACHER:-3}"
END_TEACHER="${END_TEACHER:-8}"

ROOT="$HOME/chandu_experiments"
PROJECT="$ROOT/DPI_experiments"
DATA_ROOT="$ROOT/DPI/datasets"
SPLIT_ROOT="$DATA_ROOT/BindingDB_AIBind_outer${OUTER_SET_ID}_nested_oof"
MANIFEST="$SPLIT_ROOT/nested_oof_manifest.json"
TEMPLATE_CONFIG="$PROJECT/config_auto_v37_outer4_nested_teacher00.yaml"
STATUS_FILE="$ROOT/v37_outer${OUTER_SET_ID}_remaining_teachers.status"
MASTER_LOG="$ROOT/v37_outer${OUTER_SET_ID}_remaining_teachers_failfast.log"

CURRENT_TEACHER="preflight"

write_status() {
  local state="$1"
  local detail="$2"
  local tmp="${STATUS_FILE}.tmp"
  printf '%s\t%s\tteacher=%s\t%s\n' \
    "$(date --iso-8601=seconds)" "$state" "$CURRENT_TEACHER" "$detail" > "$tmp"
  mv "$tmp" "$STATUS_FILE"
}

on_error() {
  local rc=$?
  local line="${BASH_LINENO[0]:-unknown}"
  local command="${BASH_COMMAND:-unknown}"
  write_status "FAILED" "exit=$rc line=$line command=$command"
  printf 'FATAL: teacher=%s exit=%s line=%s command=%s\n' \
    "$CURRENT_TEACHER" "$rc" "$line" "$command" >&2
  exit "$rc"
}

die() {
  local message="$1"
  local rc="${2:-2}"
  write_status "FAILED" "$message"
  echo "FATAL: $message" >&2
  exit "$rc"
}
trap on_error ERR
trap 'write_status "INTERRUPTED" "signal=INT"; exit 130' INT
trap 'write_status "INTERRUPTED" "signal=TERM"; exit 143' TERM

exec > >(tee -a "$MASTER_LOG") 2>&1
cd "$PROJECT"
write_status "RUNNING" "starting preflight"

if [[ "$OUTER_SET_ID" != "0" ]]; then
  die "This runner is approved only for outer Fold0."
fi
if (( START_TEACHER < 3 || END_TEACHER > 8 || START_TEACHER > END_TEACHER )); then
  die "Expected 3 <= START_TEACHER <= END_TEACHER <= 8."
fi

for required in \
  "$MANIFEST" \
  "$TEMPLATE_CONFIG" \
  "$PROJECT/train.py" \
  "$PROJECT/prepare_nested_teacher_config.py" \
  "$PROJECT/audit_nested_teacher_run.py"; do
  [[ -s "$required" ]] || die "Missing required file: $required"
done

python -m py_compile \
  train.py src/dataset.py src/lightning_module.py \
  prepare_nested_teacher_config.py audit_nested_teacher_run.py \
  aggregate_nested_dual_blind_oof.py build_nested_dual_blind_teacher_splits.py

python - "$MANIFEST" <<'PY'
import json
import sys

manifest = json.load(open(sys.argv[1]))
assert manifest["protocol"] == "outer-train-only-nested-dual-entity-blind-v1"
assert manifest["outer_set_id"] == 0
assert manifest["teacher_count"] == 9
assert manifest["outer_val_accessed"] is False
assert manifest["outer_test_accessed"] is False
assert manifest["target_coverage_rows"] == manifest["outer_unique_rows"]
assert all(t["target_train_drug_overlap"] == 0 for t in manifest["teachers"])
assert all(t["target_train_protein_overlap"] == 0 for t in manifest["teachers"])
print("FOLD0 MANIFEST PREFLIGHT OK")
PY

echo "=== Auditing already completed teachers 00-02 ==="
for teacher_id in 0 1 2; do
  teacher_tag="$(printf '%02d' "$teacher_id")"
  CURRENT_TEACHER="$teacher_tag"
  config_path="$PROJECT/config_auto_v37_outer0_nested_teacher${teacher_tag}.yaml"
  run_dir="$(ls -td "$PROJECT"/runs/v37_outer0_nested_teacher${teacher_tag}_* 2>/dev/null | head -1)"
  [[ -n "$run_dir" ]] || die "Missing run for teacher $teacher_tag" 3
  python audit_nested_teacher_run.py \
    --manifest "$MANIFEST" \
    --outer-set-id "$OUTER_SET_ID" \
    --teacher-id "$teacher_id" \
    --run-dir "$run_dir" \
    --config "$config_path" \
    --template-config "$TEMPLATE_CONFIG"
done

for teacher_id in $(seq "$START_TEACHER" "$END_TEACHER"); do
  teacher_tag="$(printf '%02d' "$teacher_id")"
  CURRENT_TEACHER="$teacher_tag"
  experiment_name="v37_outer0_nested_teacher${teacher_tag}"
  config_path="$PROJECT/config_auto_${experiment_name}.yaml"
  teacher_log="$ROOT/${experiment_name}.log"

  write_status "RUNNING" "preparing config"
  python prepare_nested_teacher_config.py \
    --template-config "$TEMPLATE_CONFIG" \
    --manifest "$MANIFEST" \
    --outer-set-id "$OUTER_SET_ID" \
    --teacher-id "$teacher_id" \
    --output "$config_path"

  run_dir="$(ls -td "$PROJECT"/runs/${experiment_name}_* 2>/dev/null | head -1 || true)"
  if [[ -n "$run_dir" && -s "$run_dir/test_predictions.csv" ]]; then
    echo "=== Auditing reusable teacher $teacher_tag: $run_dir ==="
  else
    if pgrep -af "train.py --config $(basename "$config_path")" >/dev/null; then
      die "A separate process is already training teacher $teacher_tag" 4
    fi
    write_status "RUNNING" "training"
    echo "=== Training teacher $teacher_tag ==="
    PYTHONUNBUFFERED=1 python train.py --config "$config_path" \
      2>&1 | tee "$teacher_log"
    run_dir="$(ls -td "$PROJECT"/runs/${experiment_name}_* 2>/dev/null | head -1)"
  fi

  [[ -n "$run_dir" ]] || die "No run directory for teacher $teacher_tag" 5
  write_status "RUNNING" "auditing run=$run_dir"
  python audit_nested_teacher_run.py \
    --manifest "$MANIFEST" \
    --outer-set-id "$OUTER_SET_ID" \
    --teacher-id "$teacher_id" \
    --run-dir "$run_dir" \
    --config "$config_path" \
    --template-config "$TEMPLATE_CONFIG"
  write_status "PASSED" "run=$run_dir"
done

CURRENT_TEACHER="all"
write_status "COMPLETE" "teachers ${START_TEACHER}-${END_TEACHER} passed"
echo "ALL REQUESTED FOLD0 TEACHERS PASSED"
