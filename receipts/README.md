# Five-Fold Run Receipts

`v37_fivefold_compact/` is the checksum-verified compact receipt bundle copied from the original experiment host on 2026-11-21.

It contains exact Stage-1 and Stage-2 configurations, metrics, all 45 nested-teacher configurations and predictions, five frozen-test prediction exports, OOF calibration products, split/manifold identities, and selected-checkpoint SHA-256 identities. Large datasets, embedding dictionaries, and checkpoint weight binaries are intentionally outside this compact bundle.

Verify the bundle and independently recompute all five reported V3.7 AUC/AUPR values with:

```bash
python scripts/audit_fivefold_receipts.py
```
