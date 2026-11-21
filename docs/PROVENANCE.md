# Provenance

## Included experiment evidence

The compact receipt tree was copied from the original experiment host on 2026-11-21. Its archive SHA-256 was:

```text
7cb19e926723f6c90f6b2950560d79644337b04be87c8ccf1866d3f3269cafbe
```

The extracted package contains 285 files: one checksum manifest plus 284 verified artifacts. These include all five Stage-1 and Stage-2 configurations and metrics, all 45 teacher prediction files, OOF/calibration outputs, all five frozen prediction exports, split identities, and selected-checkpoint identities.

`scripts/audit_fivefold_receipts.py` independently verifies that receipt tree and recomputes the five final AUC/AUPR pairs.

## Artifact boundaries

This repository stores compact code, configurations, predictions, manifests, and checksums. It does not duplicate large source datasets, packed embedding dictionaries, or checkpoint tensors. Those artifacts are produced by the documented workflow and are identified in the receipts by path, filename, size, and SHA-256 where available.

No metric or prediction was synthesized during collation, and the frozen five-fold outputs were not changed.
