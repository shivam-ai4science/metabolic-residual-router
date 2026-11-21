# Metabolic Residual Router

Clean code and reproducibility package for the five-fold blind-start CPI experiments in *Data-Ready Blind-Start CPI Evaluation through Fold-Safe Metabolic Transfer*.

Authors: Aaditya Narayan Saxena and Shivam Saxena.

## Results

The final system improves both AUC and AUPR over the matched ColdstartCPI reproduction and the preceding gated-residual system on every predefined fold.

| System | Mean AUC | Mean AUPR |
|---|---:|---:|
| Published ColdstartCPI | 0.839082 | 0.784568 |
| Matched ColdstartCPI reproduction | 0.836145 | 0.782751 |
| Gated residual | 0.838261 | 0.785212 |
| Metabolic residual router | **0.842217** | **0.792995** |

The complete fold-level table is in `results/results_five_fold.csv`.

## What is included

- `code/`: model, data preparation, training, calibration, evaluation, and audit code.
- `code/tests/`: protocol and model unit tests.
- `metabolic_corpus_notebooks/`: retained metabolic-corpus construction notebooks.
- `receipts/v37_fivefold_compact/`: exact compact evidence for all five folds, including configurations, validation metrics, nested-teacher predictions, OOF/calibration products, frozen test predictions, split identities, and checkpoint identities.
- `scripts/`: standalone five-fold receipt and benchmark-split auditors.
- `results/`: locked five-fold metrics and generated machine-readable audit reports.
- `docs/PROVENANCE.md`: provenance and artifact boundaries.

This repository deliberately excludes manuscript and hosted-release files, large embedding dictionaries, datasets, and checkpoint binaries.

## Verify the five-fold results

The primary audit uses only the Python standard library:

```bash
python scripts/audit_fivefold_receipts.py
```

It verifies the compact receipt checksums, all 45 nested-teacher targets, OOF/calibration coverage, validation-selected checkpoint identities, absence of test metrics during Stage-2 fitting, and all five frozen prediction exports. It then recomputes every reported AUC and AUPR.

Expected selected checkpoints and frozen metrics:

| Fold | Rows | AUC | AUPR | Checkpoint |
|---:|---:|---:|---:|---|
| 0 | 7,869 | 0.831547 | 0.759375 | `best_model_epoch37.ckpt` |
| 1 | 8,366 | 0.849985 | 0.808988 | `best_model_epoch37.ckpt` |
| 2 | 7,902 | 0.850309 | 0.809800 | `best_model_epoch26.ckpt` |
| 3 | 8,125 | 0.848727 | 0.808871 | `best_model_epoch38.ckpt` |
| 4 | 7,930 | 0.830515 | 0.777943 | `best_model_epoch37.ckpt` |

See `REPRODUCIBILITY.md` for the full regeneration sequence and artifact map.

## Environment

The experiment environment used Python 3.8. Install the model dependencies with:

```bash
python -m pip install -r requirements.txt
```

The metabolic-corpus notebooks have additional dependencies in `requirements-corpus.txt`.

Run tests from `code/`:

```bash
PYTHONPATH="$PWD" python -m unittest discover -s tests -p "test_*.py" -v
```

## License

No project license has been selected yet. Source datasets and pretrained features remain subject to their original terms.
