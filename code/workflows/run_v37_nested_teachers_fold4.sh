#!/usr/bin/env bash
set -euo pipefail

OUTER_SET_ID="${OUTER_SET_ID:-4}"
NUM_BUCKETS="${NUM_BUCKETS:-3}"
SEED="${SEED:-1234}"
TEACHER_EPOCHS="${TEACHER_EPOCHS:-150}"
TEACHER_PATIENCE="${TEACHER_PATIENCE:-30}"

ROOT="$HOME/chandu_experiments"
PROJECT="$ROOT/DPI_experiments"
DATA_ROOT="$ROOT/DPI/datasets"
SOURCE_DATASET="$DATA_ROOT/BindingDB_AIBind"
SPLIT_DATASET_NAME="BindingDB_AIBind_outer${OUTER_SET_ID}_nested_oof"
SPLIT_ROOT="$DATA_ROOT/$SPLIT_DATASET_NAME"
MANIFEST="$SPLIT_ROOT/nested_oof_manifest.json"
LOG_ROOT="$ROOT/v37_nested_oof_logs_outer${OUTER_SET_ID}_$(date +%Y%m%d_%H%M%S)"
REPORT_ROOT="$PROJECT/reports/v37_nested_oof_outer${OUTER_SET_ID}"

mkdir -p "$LOG_ROOT" "$REPORT_ROOT"
cd "$PROJECT"

python build_nested_dual_blind_teacher_splits.py \
  --bindingdb-dir "$SOURCE_DATASET" \
  --outer-set-id "$OUTER_SET_ID" \
  --output-root "$SPLIT_ROOT" \
  --num-buckets "$NUM_BUCKETS" \
  --teacher-val-frac 0.1 \
  --seed "$SEED" \
  > "$LOG_ROOT/split_manifest.log"

teacher_count="$(python - "$MANIFEST" <<'PY'
import json, sys
with open(sys.argv[1]) as handle:
    print(json.load(handle)["teacher_count"])
PY
)"

prediction_args=()
for teacher_id in $(seq 0 $((teacher_count - 1))); do
  teacher_tag="$(printf '%02d' "$teacher_id")"
  dataset_name="$SPLIT_DATASET_NAME/teacher_$teacher_tag"
  experiment_name="v37_outer${OUTER_SET_ID}_nested_teacher${teacher_tag}"
  config_path="config_auto_${experiment_name}.yaml"

  cat > "$config_path" <<YAML
experiment:
  name: "$experiment_name"
  mode: "pretrain"
  save_dir: "./runs"
  seed: $SEED

data:
  dataset_name: "$dataset_name"
  base_path: "../DPI/datasets"
  set_type: blind_start
  set_id: 0
  read_on_fly: false
  batch_size_train: 256
  batch_size_val: 256
  num_workers: 0
  embeddings: "coldstartcpi"
  max_lig_tokens: 101
  max_prot_tokens: 1001
  padding_mode: "official_fixed"
  official_lig_local_tokens: 100
  official_prot_local_tokens: 1000

model:
  name: "ColdstartCPIRepro"
  params:
    latent_dim: 512
    num_heads: 4
    dropout: 0.1
    use_padding_mask: false
    output_dim: 2

loss:
  name: CrossEntropyFromLogitsLoss
  use_dynamic_weights: false
  params:
    label_smoothing: 0.0

trainer:
  seed: $SEED
  max_epochs: $TEACHER_EPOCHS
  learning_rate: 1e-4
  weight_decay: 0.0
  optimizer: Adam
  accumulation_steps: 1
  monitor_metric: val_score
  monitor_mode: max
  early_stopping_patience: $TEACHER_PATIENCE
  save_every_n_epochs: 25

evaluation:
  # This is an inner OOF target block drawn only from outer train_set4.
  test_after_fit: true
  export_predictions: true
YAML

  run_dir="$(ls -td runs/${experiment_name}_* 2>/dev/null | head -1 || true)"
  if [[ -z "$run_dir" || ! -s "$run_dir/test_predictions.csv" ]]; then
    echo "=== Training nested teacher $teacher_tag/$((teacher_count - 1)) ==="
    python train.py --config "$config_path" \
      2>&1 | tee "$LOG_ROOT/teacher_${teacher_tag}.log"
    run_dir="$(ls -td runs/${experiment_name}_* | head -1)"
  else
    echo "=== Reusing completed nested teacher $teacher_tag: $run_dir ==="
  fi
  if [[ ! -s "$run_dir/test_predictions.csv" ]]; then
    echo "Missing nested OOF predictions: $run_dir/test_predictions.csv" >&2
    exit 1
  fi
  prediction_args+=(--prediction "${teacher_id}=${run_dir}/test_predictions.csv")
done

OOF_CSV="$REPORT_ROOT/BindingDB_AIBind_outer${OUTER_SET_ID}_nested_dual_blind_oof_train.csv"
OOF_SUMMARY="$REPORT_ROOT/BindingDB_AIBind_outer${OUTER_SET_ID}_nested_dual_blind_oof_summary.json"
python aggregate_nested_dual_blind_oof.py \
  --manifest "$MANIFEST" \
  "${prediction_args[@]}" \
  --output-csv "$OOF_CSV" \
  --summary-json "$OOF_SUMMARY"

ROUTER_DATASET="$DATA_ROOT/BindingDB_AIBind_router_oof_outer${OUTER_SET_ID}_nested_dual_blind"
python prepare_router_oof_dataset.py \
  --bindingdb-dir "$SOURCE_DATASET" \
  --oof-train-csv "$OOF_CSV" \
  --output-dir "$ROUTER_DATASET" \
  --set-id "$OUTER_SET_ID"

echo "=== Nested dual-blind OOF preparation complete ==="
echo "Manifest: $MANIFEST"
echo "OOF targets: $OOF_CSV"
echo "Router dataset: $ROUTER_DATASET"
echo "Logs: $LOG_ROOT"
