#!/usr/bin/env bash
set -euo pipefail
trap 'status=$?; echo "ERROR: exit=${status} line=${LINENO} command=${BASH_COMMAND}" >&2' ERR

# Historical workflow default covers folds 0, 1, and 2.
# Override at launch, for example:
#   FOLDS="0 1 2 3" BASE_EPOCHS=150 bash run_v36_all_remaining_folds.sh
FOLDS="${FOLDS:-0 1 2}"
BASE_EPOCHS="${BASE_EPOCHS:-150}"
BASE_PATIENCE="${BASE_PATIENCE:-50}"
SEED="${SEED:-1234}"
CLEANUP_METABOLIC="${CLEANUP_METABOLIC:-1}"
PREFLIGHT_ONLY="${PREFLIGHT_ONLY:-0}"

ROOT="$HOME/chandu_experiments"
DPI_EXP="$ROOT/DPI_experiments"
DPI_DATA="$ROOT/DPI/datasets"
COLD_ROOT="$ROOT/ColdstartCPI-main"
COLD_DATA="$COLD_ROOT/Datasets"
LOG_DIR="$ROOT/fold_logs_v36_$(date +%Y%m%d_%H%M%S)"
SUMMARY="$LOG_DIR/summary.tsv"

mkdir -p "$LOG_DIR"
cd "$DPI_EXP"

echo -e "fold\tbaseline_run\tadapter_run\tfixed_alpha_run\tbaseline_test_score\tfixed_test_score" > "$SUMMARY"

latest_run() {
  local pattern="$1"
  local matches=()
  local match
  while IFS= read -r match; do
    matches+=("$match")
  done < <(compgen -G "$pattern" || true)
  if (( ${#matches[@]} == 0 )); then
    return 0
  fi
  ls -td -- "${matches[@]}" 2>/dev/null | head -1 || true
}

latest_ckpt() {
  local run_dir="$1"
  local matches=()
  local match
  if [[ -z "$run_dir" ]]; then
    return 0
  fi
  while IFS= read -r match; do
    matches+=("$match")
  done < <(compgen -G "$run_dir/best_model_epoch*.ckpt" || true)
  if (( ${#matches[@]} == 0 )); then
    return 0
  fi
  ls -t -- "${matches[@]}" 2>/dev/null | head -1 || true
}

require_file() {
  local path="$1"
  if [[ ! -e "$path" ]]; then
    echo "Missing required file/path: $path" >&2
    exit 1
  fi
}

patch_feature_generator_dataset_choices() {
  echo "=== Making ColdstartCPI feature generators accept fold-specific dataset names (strict parser patch v2) ==="
  python - <<'PY'
from pathlib import Path
import re

paths = [
    Path.home() / "chandu_experiments/ColdstartCPI-main/Feature_generation/Mol2Vec/Mol2Vec.py",
    Path.home() / "chandu_experiments/ColdstartCPI-main/Feature_generation/ProtTrans/generator.py",
]


def paren_block_end(lines, start):
    balance = 0
    end = start
    for j in range(start, len(lines)):
        balance += lines[j].count("(") - lines[j].count(")")
        end = j
        if balance <= 0:
            break
    return end


def strip_dataset_argument_blocks(lines):
    stripped = []
    i = 0
    while i < len(lines):
        line = lines[i]
        if "add_argument" in line:
            end = paren_block_end(lines, i)
            block = "".join(lines[i : end + 1])
            if "--dataset" in block:
                i = end + 1
                continue
        stripped.append(line)
        i += 1
    return stripped


def replace_dataset_argument(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(path)

    backup = path.with_suffix(path.suffix + ".dataset_choices.bak")
    current_text = path.read_text()
    if backup.exists():
        source_text = backup.read_text()
        print(f"restoring clean source from {backup}")
    else:
        backup.write_text(current_text)
        source_text = current_text

    lines = strip_dataset_argument_blocks(source_text.splitlines(keepends=True))

    parser_idx = None
    parser_var = None
    parser_assignment = re.compile(r"^(\s*)([A-Za-z_][A-Za-z0-9_]*)\s*=\s*argparse\.ArgumentParser\b")
    for i, line in enumerate(lines):
        match = parser_assignment.search(line)
        if match:
            parser_idx = i
            parser_var = match.group(2)
            break
    if parser_idx is None:
        raise RuntimeError(f"could not find a parser = argparse.ArgumentParser(...) assignment in {path}")

    parser_end = paren_block_end(lines, parser_idx)
    indent = lines[parser_idx][: len(lines[parser_idx]) - len(lines[parser_idx].lstrip())]
    replacement = f'{indent}{parser_var}.add_argument("--dataset", default="BindingDB_AIBind")\n'
    new_lines = lines[: parser_end + 1] + [replacement] + lines[parser_end + 1 :]
    dataset_idx = next(
        i for i, line in enumerate(new_lines)
        if f'{parser_var}.add_argument("--dataset"' in line
    )
    if dataset_idx <= parser_idx:
        raise RuntimeError(f"dataset argument was inserted before parser creation in {path}")
    path.write_text("".join(new_lines))
    print(f"patched {path} parser_line={parser_idx + 1} dataset_line={dataset_idx + 1}")


for path in paths:
    replace_dataset_argument(path)
PY
}

write_baseline_config() {
  local fold="$1"
  local cfg="config_auto_aibind_coldstart_repro_ce_fold${fold}_${BASE_EPOCHS}ep.yaml"
  cat > "$cfg" <<YAML
experiment:
  name: "aibind_coldstart_repro_ce_fold${fold}_${BASE_EPOCHS}ep"
  mode: "pretrain"
  save_dir: "./runs"
  seed: ${SEED}

data:
  dataset_name: "BindingDB_AIBind"
  base_path: "../DPI/datasets"
  set_type: blind_start
  set_id: ${fold}
  read_on_fly: false
  batch_size_train: 256
  batch_size_val: 256
  num_workers: 0
  embeddings: "coldstartcpi"
  max_lig_tokens: 101
  max_prot_tokens: 1001

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
  seed: ${SEED}
  max_epochs: ${BASE_EPOCHS}
  learning_rate: 1e-4
  weight_decay: 0.0
  optimizer: Adam
  accumulation_steps: 1
  monitor_metric: val_score
  monitor_mode: max
  early_stopping_patience: ${BASE_PATIENCE}
  save_every_n_epochs: 25

evaluation:
  test_after_fit: true
  export_predictions: true
YAML
  echo "$cfg"
}

write_adapter_config() {
  local fold="$1"
  local baseline_run="$2"
  local metabolic_dataset="$3"
  local cfg="config_auto_metabolic_pathway_gated_adapter_fold${fold}_v36_pu_rank.yaml"
  cat > "$cfg" <<YAML
experiment:
  name: "metabolic_pathway_gated_adapter_fold${fold}_v36_pu_rank"
  mode: "finetune"
  save_dir: "./runs"
  seed: ${SEED}

finetune:
  source_run: "${baseline_run}"
  checkpoint_type: "best"
  freeze_backbone: false

data:
  dataset_name: "${metabolic_dataset}"
  base_path: "../DPI/datasets"
  set_type: blind_start
  set_id: 0
  read_on_fly: false
  batch_size_train: 64
  batch_size_val: 128
  num_workers: 0
  embeddings: "coldstartcpi"
  max_lig_tokens: 101
  max_prot_tokens: 1001
  train_fraction: 0.5
  val_fraction: 0.5

model:
  name: "PathwayGatedColdstartCPI"
  params:
    latent_dim: 512
    num_heads: 4
    dropout: 0.15
    use_padding_mask: false
    output_dim: 2
    topk: 32
    gate_bias: -1.5
    max_gate: 0.50
    max_delta: 4.0
    delta_init_std: 0.006
    freeze_base: true
    detach_base_features: true

loss:
  name: PathwayResidualPULoss
  use_dynamic_weights: false
  params:
    pairwise_weight: 1.0
    pairwise_margin: 0.20
    pos_margin_weight: 0.15
    pos_margin: 0.10
    unlabeled_positive_penalty_weight: 0.04
    unlabeled_margin: 0.25
    center_weight: 0.20
    gate_l1_weight: 0.004
    correction_l1_weight: 0.004
    delta_l2_weight: 0.0001
    entropy_weight: 0.001

trainer:
  seed: ${SEED}
  max_epochs: 14
  learning_rate: 8e-4
  weight_decay: 1e-4
  optimizer: AdamW
  accumulation_steps: 2
  monitor_metric: val_correction_score
  monitor_mode: max
  early_stopping_patience: 5
  save_every_n_epochs: 4

evaluation:
  test_after_fit: false
  export_predictions: false
YAML
  echo "$cfg"
}

write_fixed_alpha_config() {
  local fold="$1"
  local adapter_run="$2"
  local cfg="config_auto_aibind_pathway_gated_fold${fold}_v36_fixed_alpha1.yaml"
  cat > "$cfg" <<YAML
experiment:
  name: "aibind_pathway_gated_fold${fold}_v36_fixed_alpha1"
  mode: "finetune"
  save_dir: "./runs"
  seed: ${SEED}

finetune:
  source_run: "${adapter_run}"
  checkpoint_type: "best"
  freeze_backbone: false

data:
  dataset_name: "BindingDB_AIBind"
  base_path: "../DPI/datasets"
  set_type: blind_start
  set_id: ${fold}
  read_on_fly: false
  batch_size_train: 256
  batch_size_val: 256
  num_workers: 0
  embeddings: "coldstartcpi"
  max_lig_tokens: 101
  max_prot_tokens: 1001

model:
  name: "PathwayGatedColdstartCPI"
  params:
    latent_dim: 512
    num_heads: 4
    dropout: 0.1
    use_padding_mask: false
    output_dim: 2
    topk: 32
    gate_bias: -1.5
    max_gate: 0.50
    max_delta: 4.0
    delta_init_std: 0.0
    freeze_base: true
    detach_base_features: true
    freeze_pathway_adapter: true
    train_correction_scale: true
    correction_scale_init: 1.0
    correction_scale_max: 2.0
    use_base_uncertainty_gate: false

loss:
  name: GatedResidualCrossEntropyLoss
  use_dynamic_weights: false
  params:
    label_smoothing: 0.0
    gate_l1_weight: 0.0
    delta_l2_weight: 0.0
    correction_l1_weight: 0.0
    negative_positive_correction_weight: 0.0
    positive_negative_correction_weight: 0.0
    entropy_weight: 0.0
    distill_weight: 0.0

trainer:
  seed: ${SEED}
  max_epochs: 1
  learning_rate: 0.0
  weight_decay: 0.0
  optimizer: Adam
  accumulation_steps: 1
  monitor_metric: val_score
  monitor_mode: max
  early_stopping_patience: 1
  save_every_n_epochs: 0

evaluation:
  test_after_fit: true
  export_predictions: true
YAML
  echo "$cfg"
}

prepare_fold_safe_metabolic_dataset() {
  local fold="$1"
  local strict="metabolic_pathway_phase_2_fold${fold}_strict"
  local random_pair="${strict}_random_pair_split"
  local cold_dataset="$COLD_DATA/$strict"
  local dpi_dataset="$DPI_DATA/$strict"
  local available_kb
  local required_kb

  if [[ ! -s "$DPI_DATA/$random_pair/embeddings/drugs/ligand_embeddings_coldstartcpi.pt" ||
        ! -s "$DPI_DATA/$random_pair/embeddings/proteins_prottrans_bfd/protein_embeddings_coldstartcpi.pt" ]]; then
    echo "=== Building fold-safe metabolic dataset for fold ${fold}: ${strict} ==="
    python build_fold_safe_metabolic_dataset.py \
      --bindingdb-dir "$DPI_DATA/BindingDB_AIBind" \
      --metabolic-path "$DPI_DATA/BindingDB_AIBind/metabolic_pathway_phase_2.zip" \
      --set-id "$fold" \
      --output-dataset-dir "$DPI_DATA/$strict"

    python make_metabolic_diagnostic_splits.py \
      --source-dataset-dir "$DPI_DATA/$strict" \
      --output-dataset-dir "$DPI_DATA/$random_pair"

    mkdir -p "$COLD_DATA"
    if [[ -L "$cold_dataset" ]]; then
      if [[ "$(readlink -f "$cold_dataset")" != "$(readlink -f "$dpi_dataset")" ]]; then
        echo "Refusing mismatched dataset symlink: $cold_dataset" >&2
        exit 1
      fi
    elif [[ -e "$cold_dataset" ]]; then
      echo "Refusing to duplicate fold data: $cold_dataset already exists and is not a symlink." >&2
      echo "Move or remove that generated directory, then rerun so it can link to $dpi_dataset." >&2
      exit 1
    else
      ln -s "$dpi_dataset" "$cold_dataset"
    fi

    if [[ ! -s "$COLD_DATA/$strict/feature/compound_Mol2Vec300.pkl" ||
          ! -s "$COLD_DATA/$strict/feature/compound_Atom2Vec300.pkl" ]]; then
      echo "=== Generating Mol2Vec/Atom2Vec for ${strict} ==="
      (cd "$COLD_ROOT/Feature_generation/Mol2Vec" && python Mol2Vec.py --dataset "$strict") \
        2>&1 | tee "$LOG_DIR/fold${fold}_mol2vec.log"
    fi

    if [[ ! -s "$COLD_DATA/$strict/feature/aas_ProtTransBertBFD1024.pkl" ]]; then
      available_kb="$(df --output=avail -k "$DPI_DATA" | tail -1 | tr -d ' ')"
      required_kb=$((160 * 1024 * 1024))
      if (( available_kb < required_kb )); then
        echo "Insufficient disk before ProtTrans: need at least 160 GiB free, have $((available_kb / 1024 / 1024)) GiB." >&2
        exit 1
      fi
      echo "=== Generating ProtTrans for ${strict} ==="
      (cd "$COLD_ROOT/Feature_generation/ProtTrans" && python generator.py --dataset "$strict") \
        2>&1 | tee "$LOG_DIR/fold${fold}_prottrans.log"
    fi

    echo "=== Packing ColdstartCPI tensors for ${strict} ==="
    available_kb="$(df --output=avail -k "$DPI_DATA" | tail -1 | tr -d ' ')"
    required_kb=$((85 * 1024 * 1024))
    if (( available_kb < required_kb )); then
      echo "Insufficient disk for packing: need at least 85 GiB free, have $((available_kb / 1024 / 1024)) GiB." >&2
      exit 1
    fi
    python prepare_coldstartcpi_embeddings.py --dataset-dir "$DPI_DATA/$strict" \
      2>&1 | tee "$LOG_DIR/fold${fold}_pack_embeddings.log"

    require_file "$DPI_DATA/$strict/embeddings/proteins_prottrans_bfd/protein_embeddings_coldstartcpi.pt"
    echo "=== Removing recoverable raw ProtTrans pickle after successful packing ==="
    rm -f "$DPI_DATA/$strict/feature/aas_ProtTransBertBFD1024.pkl"

    if [[ -L "$DPI_DATA/$random_pair/embeddings" ]]; then
      unlink "$DPI_DATA/$random_pair/embeddings"
    elif [[ -e "$DPI_DATA/$random_pair/embeddings" ]]; then
      echo "Refusing to replace non-symlink embeddings path: $DPI_DATA/$random_pair/embeddings" >&2
      exit 1
    fi
    ln -s "../$strict/embeddings" "$DPI_DATA/$random_pair/embeddings"
  fi

  require_file "$DPI_DATA/$random_pair/embeddings/drugs/ligand_embeddings_coldstartcpi.pt"
  require_file "$DPI_DATA/$random_pair/embeddings/proteins_prottrans_bfd/protein_embeddings_coldstartcpi.pt"
  METABOLIC_DATASET="$random_pair"
  METABOLIC_STRICT_DATASET="$strict"
}

cleanup_fold_metabolic_dataset() {
  local fold="$1"
  local strict="metabolic_pathway_phase_2_fold${fold}_strict"
  local random_pair="${strict}_random_pair_split"

  if [[ "$CLEANUP_METABOLIC" != "1" ]]; then
    return
  fi

  echo "=== Cleaning generated metabolic datasets for fold ${fold} to save disk ==="
  rm -rf "$DPI_DATA/$random_pair" "$DPI_DATA/$strict" "$COLD_DATA/$strict"
}

extract_test_score() {
  local run_dir="$1"
  python - "$run_dir" <<'PY'
import os
import sys
import pandas as pd

run = sys.argv[1]
p = os.path.join(run, "version_0", "metrics.csv")
if not os.path.exists(p):
    print("NA")
    raise SystemExit
df = pd.read_csv(p)
test_cols = [c for c in ["test_auc", "test_aupr", "test_score"] if c in df.columns]
if not test_cols:
    print("NA")
    raise SystemExit
t = df[df[test_cols].notna().any(axis=1)]
if len(t) == 0:
    print("NA")
else:
    row = t.iloc[-1]
    print(f"{row.get('test_score', float('nan')):.6f}")
PY
}

patch_feature_generator_dataset_choices

python -m py_compile \
  train.py \
  src/models/pathway_gated_coldstart_cpi.py \
  src/models/__init__.py \
  src/losses/loss_funk.py \
  src/lightning_module.py \
  build_fold_safe_metabolic_dataset.py \
  make_metabolic_diagnostic_splits.py \
  prepare_coldstartcpi_embeddings.py \
  threshold_matched_error_analysis.py \
  analyze_pathway_adapter.py \
  "$COLD_ROOT/Feature_generation/Mol2Vec/Mol2Vec.py" \
  "$COLD_ROOT/Feature_generation/ProtTrans/generator.py"

echo "=== Runtime help sanity for feature generators ==="
(cd "$COLD_ROOT/Feature_generation/Mol2Vec" && python Mol2Vec.py --help >/dev/null)
(cd "$COLD_ROOT/Feature_generation/ProtTrans" && python generator.py --help >/dev/null)

require_file "$DPI_DATA/BindingDB_AIBind/metabolic_pathway_phase_2.zip"
require_file "$DPI_DATA/BindingDB_AIBind/embeddings/drugs/ligand_embeddings_coldstartcpi.pt"
require_file "$DPI_DATA/BindingDB_AIBind/embeddings/proteins_prottrans_bfd/protein_embeddings_coldstartcpi.pt"

if [[ "$PREFLIGHT_ONLY" == "1" ]]; then
  echo "=== PREFLIGHT OK ==="
  echo "Feature generators patched, Python files compile, and required BindingDB/metabolic inputs exist."
  exit 0
fi

for fold in $FOLDS; do
  echo
  echo "################################################################"
  echo "### FOLD ${fold} | BASE_EPOCHS=${BASE_EPOCHS}"
  echo "################################################################"
  nvidia-smi || true
  free -h || true
  df -h "$ROOT" || true

  baseline_run="$(latest_run "runs/aibind_coldstart_repro_ce_fold${fold}_${BASE_EPOCHS}ep_*")"
  if [[ -n "$baseline_run" && -s "$baseline_run/test_predictions.csv" ]]; then
    echo "=== Reusing existing baseline fold ${fold}: ${baseline_run} ==="
  else
    baseline_cfg="$(write_baseline_config "$fold")"
    echo "=== Baseline training fold ${fold}: ${baseline_cfg} ==="
    python train.py --config "$baseline_cfg" 2>&1 | tee "$LOG_DIR/fold${fold}_baseline.log"
    baseline_run="$(latest_run "runs/aibind_coldstart_repro_ce_fold${fold}_${BASE_EPOCHS}ep_*")"
  fi
  require_file "$baseline_run/test_predictions.csv"
  baseline_run_name="$(basename "$baseline_run")"
  echo "BASELINE_RUN=$baseline_run_name"

  prepare_fold_safe_metabolic_dataset "$fold"
  metabolic_dataset="$METABOLIC_DATASET"
  echo "METABOLIC_DATASET=$metabolic_dataset"

  adapter_run="$(latest_run "runs/metabolic_pathway_gated_adapter_fold${fold}_v36_pu_rank_*")"
  adapter_ckpt="$(latest_ckpt "$adapter_run")"
  if [[ -n "$adapter_run" && -n "$adapter_ckpt" ]]; then
    echo "=== Reusing existing adapter fold ${fold}: ${adapter_run} ==="
  else
    adapter_cfg="$(write_adapter_config "$fold" "$baseline_run_name" "$metabolic_dataset")"
    echo "=== Metabolic adapter training fold ${fold}: ${adapter_cfg} ==="
    python train.py --config "$adapter_cfg" 2>&1 | tee "$LOG_DIR/fold${fold}_adapter.log"
    adapter_run="$(latest_run "runs/metabolic_pathway_gated_adapter_fold${fold}_v36_pu_rank_*")"
    adapter_ckpt="$(latest_ckpt "$adapter_run")"
  fi
  require_file "$adapter_ckpt"
  adapter_run_name="$(basename "$adapter_run")"
  echo "ADAPTER_RUN=$adapter_run_name"

  fixed_run="$(latest_run "runs/aibind_pathway_gated_fold${fold}_v36_fixed_alpha1_*")"
  if [[ -n "$fixed_run" && -s "$fixed_run/test_predictions.csv" ]]; then
    echo "=== Reusing existing fixed-alpha fold ${fold}: ${fixed_run} ==="
  else
    fixed_cfg="$(write_fixed_alpha_config "$fold" "$adapter_run_name")"
    echo "=== Fixed alpha=1 BindingDB evaluation fold ${fold}: ${fixed_cfg} ==="
    python train.py --config "$fixed_cfg" 2>&1 | tee "$LOG_DIR/fold${fold}_fixed_alpha1.log"
    fixed_run="$(latest_run "runs/aibind_pathway_gated_fold${fold}_v36_fixed_alpha1_*")"
  fi
  require_file "$fixed_run/test_predictions.csv"
  fixed_run_name="$(basename "$fixed_run")"
  echo "FIXED_ALPHA_RUN=$fixed_run_name"

  out_dir="$fixed_run/threshold_matched_vs_baseline_test"
  python threshold_matched_error_analysis.py \
    --baseline-predictions "$baseline_run/test_predictions.csv" \
    --candidate-predictions "$fixed_run/test_predictions.csv" \
    --bindingdb-dir "$DPI_DATA/BindingDB_AIBind" \
    --output-dir "$out_dir" \
    2>&1 | tee "$LOG_DIR/fold${fold}_threshold_analysis.log"

  python analyze_pathway_adapter.py \
    --run-dir "$fixed_run" \
    --split test \
    --checkpoint best \
    --output "$fixed_run/test_pathway_adapter_analysis_best.csv" \
    --device cuda \
    2>&1 | tee "$LOG_DIR/fold${fold}_adapter_analysis.log"

  baseline_score="$(extract_test_score "$baseline_run")"
  fixed_score="$(extract_test_score "$fixed_run")"
  echo -e "${fold}\t${baseline_run_name}\t${adapter_run_name}\t${fixed_run_name}\t${baseline_score}\t${fixed_score}" >> "$SUMMARY"

  cleanup_fold_metabolic_dataset "$fold"
done

echo
echo "=== DONE ==="
cat "$SUMMARY"
echo "Logs and summary saved in: $LOG_DIR"
