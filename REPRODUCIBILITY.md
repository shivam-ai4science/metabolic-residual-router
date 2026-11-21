# Reproducibility

This repository supports two complementary reproduction paths.

## 1. Verify the reported five-fold results

All compact run evidence required to verify the final metrics is versioned under `receipts/v37_fivefold_compact/`.

Run:

```bash
python scripts/audit_fivefold_receipts.py
```

The audit fails unless all of the following hold:

1. Every receipt file matches the package-level SHA-256 manifest.
2. Folds 0--4 contain the expected Stage-1, Stage-2, and 45 nested-teacher checkpoint identities.
3. Nested manifests record no access to the outer validation or test partitions.
4. Each teacher target has zero compound and protein overlap with its teacher-training rows.
5. The nine teacher targets cover every eligible training row exactly once within each outer fold.
6. Raw and calibrated OOF row multisets match the teacher targets.
7. The selected Stage-2 checkpoint matches the best monitored validation epoch.
8. Stage-2 training configurations set `test_after_fit: false`, and their training metrics contain no test evaluation.
9. Each frozen evaluated cohort has the locked row count.
10. AUC and AUPR recomputed from every frozen score export match `results/results_five_fold.csv`.

The audit writes `results/fivefold_receipt_audit.json`.

## 2. Regenerate the training pipeline

The full pipeline is defined by the retained code plus exact historical configurations. Place the benchmark and metabolic source data in the locations configured for your environment, then run the stages below independently for each outer fold.

### Stage A: fold-safe metabolic data

1. Build the fold-filtered metabolic dataset with `code/build_fold_safe_metabolic_dataset.py`.
2. Build its diagnostic split with `code/make_metabolic_diagnostic_splits.py`.
3. Prepare Mol2Vec/Atom2Vec and ProtTrans inputs with `code/prepare_coldstartcpi_embeddings.py`; `code/reuse_coldstartcpi_embeddings_by_value.py` can reuse exact-value embeddings while auditing coverage.
4. Train the side representation with `code/train.py` and `receipts/v37_fivefold_compact/configs/stage1_fold{0..4}.yaml`.

### Stage B: nested dual-entity-blind targets

1. Build nine nested teacher partitions with `code/build_nested_dual_blind_teacher_splits.py`.
2. Generate teacher configurations with `code/prepare_nested_teacher_config.py` and train each teacher with `code/train.py`.
3. Audit every teacher with `code/audit_nested_teacher_run.py`.
4. Aggregate OOF predictions with `code/aggregate_nested_dual_blind_oof.py`.
5. Cross-fit calibrate the targets with `code/calibrate_nested_oof_targets.py`.

The exact historical teacher configurations, metrics, and target predictions are retained under `receipts/v37_fivefold_compact/teachers/`.

### Stage C: frozen-baseline residual routing

1. Build the router dataset with `code/prepare_router_oof_dataset.py`.
2. Generate the Stage-2 configuration with `code/build_v37_stage2_config.py`.
3. Run the fail-closed audit in `code/audit_v37_stage2_preflight.py`.
4. Train with the corresponding `receipts/v37_fivefold_compact/configs/stage2_fold{0..4}.yaml`.
5. Select the validation-best checkpoint before opening the held-out test partition.
6. Export frozen test scores and compare the exact evaluated row multiset.

## Artifact map

The compact receipts include:

- exact Stage-1 and Stage-2 configurations and validation metrics for all five folds;
- effective configurations, metrics, and predictions for all 45 nested teachers;
- nested split manifests and OOF/calibration outputs;
- frozen test prediction exports and summaries for all five folds;
- filename, byte-size, and SHA-256 identities for the selected checkpoints; and
- SHA-256 identities for the benchmark train/validation/test splits.

Large datasets, embedding dictionaries, and checkpoint tensors are generated inputs/outputs rather than Git artifacts. Their identities and the exact configurations that consumed them are retained in the compact receipts.

## Optional benchmark-split audit

After placing the official split files at `DPI/datasets/BindingDB_AIBind/blind_start/`, run:

```bash
python scripts/audit_final_5fold_local.py
```

This checks split hashes, double-cold train/test separation, exact-row deduplication, feature exclusions, and evaluated cohort sizes.
