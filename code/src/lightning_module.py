import os

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from sklearn import metrics


def binary_accuracy(labels, preds):
    return float((labels == preds).mean()) if labels.size else 0.0


def binary_f1(labels, preds):
    tp = float(((labels == 1) & (preds == 1)).sum())
    fp = float(((labels == 0) & (preds == 1)).sum())
    fn = float(((labels == 1) & (preds == 0)).sum())
    denom = 2 * tp + fp + fn
    return 0.0 if denom == 0 else (2 * tp) / denom


def roc_auc(labels, probs):
    labels = labels.astype(int)
    pos = labels.sum()
    neg = labels.size - pos
    if pos == 0 or neg == 0:
        return 0.5
    fpr, tpr, _ = metrics.roc_curve(labels, probs)
    return float(metrics.auc(fpr, tpr))


def pr_auc(labels, probs):
    labels = labels.astype(int)
    pos = labels.sum()
    if pos == 0:
        return 0.0
    precision, recall, _ = metrics.precision_recall_curve(labels, probs)
    return float(metrics.auc(recall, precision))


def average_precision(labels, probs):
    labels = labels.astype(int)
    pos = labels.sum()
    if pos == 0:
        return 0.0
    return float(metrics.average_precision_score(labels, probs))


def expected_calibration_error(labels, probs, num_bins=15):
    if labels.size == 0:
        return 0.0
    edges = np.linspace(0.0, 1.0, num_bins + 1)
    error = 0.0
    for index in range(num_bins):
        lower = edges[index]
        upper = edges[index + 1]
        if index == num_bins - 1:
            mask = (probs >= lower) & (probs <= upper)
        else:
            mask = (probs >= lower) & (probs < upper)
        if mask.any():
            error += mask.mean() * abs(probs[mask].mean() - labels[mask].mean())
    return float(error)


class DPILightningWrapper(pl.LightningModule):
    def __init__(self, model, loss_fn, config):
        super().__init__()
        self.model = model
        self.loss_fn = loss_fn
        self.config = config

        self.train_step_outputs = []
        self.val_step_outputs = []
        self.test_step_outputs = []

    def forward(self, prot, lig, lig_mask, prot_mask=None, router_teacher_probability=None):
        if router_teacher_probability is not None:
            return self.model(
                prot,
                lig,
                lig_mask,
                prot_mask,
                router_teacher_probability=router_teacher_probability,
            )
        try:
            return self.model(prot, lig, lig_mask, prot_mask)
        except TypeError:
            return self.model(prot, lig, lig_mask)

    def _unpack_batch(self, batch):
        if len(batch) == 7:
            prot, lig, prot_mask, lig_mask, labels, weights, metas = batch
            return prot, lig, prot_mask, lig_mask, labels, weights, metas
        if len(batch) == 6:
            prot, lig, prot_mask, lig_mask, labels, metas = batch
            return prot, lig, prot_mask, lig_mask, labels, None, metas
        if len(batch) == 5:
            prot, lig, lig_mask, labels, metas = batch
            return prot, lig, None, lig_mask, labels, None, metas

        prot, lig, lig_mask, labels = batch
        return prot, lig, None, lig_mask, labels, None, None

    def _compute_loss(self, outputs, labels):
        if getattr(self.loss_fn, "expects_outputs_dict", False):
            return self.loss_fn(outputs, labels)
        logits = outputs["logits"] if isinstance(outputs, dict) else outputs
        return self.loss_fn(logits, labels)

    @staticmethod
    def _meta_tensor(metas, key, device):
        if metas and all(meta is not None and key in meta for meta in metas):
            return torch.tensor(
                [meta[key] for meta in metas],
                device=device,
                dtype=torch.float32,
            )
        return None

    @classmethod
    def _attach_meta_targets(cls, outputs, metas, device):
        if not metas:
            return
        for key in ["oof_base_prob", "oof_base_logit", "oof_teacher_count", "oof_teacher_std"]:
            value = cls._meta_tensor(metas, key, device)
            if value is not None:
                outputs[key] = value

    def on_train_epoch_start(self):
        epoch = self.current_epoch
        beta_start = self.config["trainer"].get("beta_start", 2.0)
        beta_end = self.config["trainer"].get("beta_end", 2.0)
        anneal_epochs = self.config["trainer"].get("beta_anneal_epochs", 50)

        current_beta = beta_start if epoch < anneal_epochs else beta_end
        if hasattr(self.model, "beta"):
            self.model.beta = current_beta

        self.log("beta", current_beta, on_step=False, on_epoch=True, prog_bar=True)

    def training_step(self, batch, batch_idx):
        prot, lig, prot_mask, lig_mask, labels, weights, metas = self._unpack_batch(batch)
        oof_base_prob = self._meta_tensor(metas, "oof_base_prob", labels.device)
        outputs = self(
            prot,
            lig,
            lig_mask,
            prot_mask,
            router_teacher_probability=oof_base_prob,
        )
        outputs["sample_weight"] = weights
        self._attach_meta_targets(outputs, metas, labels.device)
        logits = outputs["logits"]
        loss = self._compute_loss(outputs, labels)

        if torch.isnan(loss) or torch.isinf(loss) or torch.isnan(logits).any():
            print(f"\nNaN/Inf detected at training step {batch_idx}")
            nan_mask = torch.isnan(logits)
            if nan_mask.any():
                for b_idx in torch.where(nan_mask)[0].unique():
                    meta = metas[b_idx] if metas else {}
                    print(f"Batch item {int(b_idx)} original row: {meta.get('row_index', int(b_idx))}")
            raise ValueError("Training halted due to NaN/Inf.")

        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self._log_adapter_stats(outputs, "train")
        self.train_step_outputs.append({"logits": logits.detach(), "labels": labels.detach()})
        return loss

    def validation_step(self, batch, batch_idx):
        prot, lig, prot_mask, lig_mask, labels, weights, metas = self._unpack_batch(batch)
        outputs = self(prot, lig, lig_mask, prot_mask)
        outputs["sample_weight"] = weights
        self._attach_meta_targets(outputs, metas, labels.device)
        logits = outputs["logits"]
        loss = self._compute_loss(outputs, labels)

        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self._log_adapter_stats(outputs, "val")
        val_output = {"logits": logits.detach(), "labels": labels.detach()}
        for key in ["pathway_correction", "base_logits", "metabolic_compatibility_logit"]:
            value = outputs.get(key) if isinstance(outputs, dict) else None
            if value is not None:
                val_output[key] = value.detach()
        self.val_step_outputs.append(val_output)
        return loss

    def test_step(self, batch, batch_idx):
        prot, lig, prot_mask, lig_mask, labels, weights, metas = self._unpack_batch(batch)
        outputs = self(prot, lig, lig_mask, prot_mask)
        outputs["sample_weight"] = weights
        self._attach_meta_targets(outputs, metas, labels.device)
        logits = outputs["logits"]
        loss = self._compute_loss(outputs, labels)

        self.log("test_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        test_output = {
            "logits": logits.detach(),
            "labels": labels.detach(),
            "metas": metas,
        }
        for key in [
            "base_logits",
            "metabolic_compatibility_logit",
            "metabolic_relevance",
            "router_rescue_probability",
            "router_suppress_probability",
            "router_abstain_probability",
            "rescue_magnitude",
            "suppress_magnitude",
            "pathway_base_uncertainty",
            "pathway_correction",
        ]:
            value = outputs.get(key) if isinstance(outputs, dict) else None
            if value is not None:
                test_output[key] = value.detach()
        self.test_step_outputs.append(test_output)
        return loss

    def _log_adapter_stats(self, outputs, prefix):
        gate = outputs.get("pathway_gate") if isinstance(outputs, dict) else None
        correction = outputs.get("pathway_correction") if isinstance(outputs, dict) else None
        delta = outputs.get("pathway_delta") if isinstance(outputs, dict) else None
        entropy = outputs.get("pathway_contact_entropy") if isinstance(outputs, dict) else None
        relevance = outputs.get("metabolic_relevance") if isinstance(outputs, dict) else None
        rescue_probability = outputs.get("router_rescue_probability") if isinstance(outputs, dict) else None
        suppress_probability = outputs.get("router_suppress_probability") if isinstance(outputs, dict) else None
        abstain_probability = outputs.get("router_abstain_probability") if isinstance(outputs, dict) else None

        if gate is not None:
            self.log(f"{prefix}_pathway_gate", gate.detach().mean(), on_step=False, on_epoch=True, prog_bar=False)
        if correction is not None:
            self.log(
                f"{prefix}_pathway_abs_correction",
                correction.detach().abs().mean(),
                on_step=False,
                on_epoch=True,
                prog_bar=False,
            )
            self.log(
                f"{prefix}_pathway_mean_correction",
                correction.detach().mean(),
                on_step=False,
                on_epoch=True,
                prog_bar=False,
            )
        if delta is not None:
            self.log(f"{prefix}_pathway_abs_delta", delta.detach().abs().mean(), on_step=False, on_epoch=True, prog_bar=False)
        if entropy is not None:
            self.log(f"{prefix}_pathway_entropy", entropy.detach().mean(), on_step=False, on_epoch=True, prog_bar=False)
        if relevance is not None:
            self.log(f"{prefix}_metabolic_relevance", relevance.detach().mean(), on_step=False, on_epoch=True, prog_bar=False)
        if rescue_probability is not None:
            self.log(f"{prefix}_router_rescue", rescue_probability.detach().mean(), on_step=False, on_epoch=True, prog_bar=False)
        if suppress_probability is not None:
            self.log(f"{prefix}_router_suppress", suppress_probability.detach().mean(), on_step=False, on_epoch=True, prog_bar=False)
        if abstain_probability is not None:
            self.log(f"{prefix}_router_abstain", abstain_probability.detach().mean(), on_step=False, on_epoch=True, prog_bar=False)

    def calculate_metrics(self, outputs_list):
        if not outputs_list:
            empty = np.array([])
            return 0.5, 0.0, 0.0, 0.0, empty, empty, empty

        raw_logits = torch.cat([x["logits"] for x in outputs_list])
        labels = torch.cat([x["labels"] for x in outputs_list]).cpu().numpy()
        if raw_logits.ndim == 2 and raw_logits.size(1) == 2:
            probs_tensor = torch.softmax(raw_logits, dim=1)[:, 1]
            preds_tensor = torch.argmax(raw_logits, dim=1)
            logits = (raw_logits[:, 1] - raw_logits[:, 0]).cpu().numpy()
            probs = probs_tensor.cpu().numpy()
            preds = preds_tensor.cpu().numpy().astype(int)
        else:
            logits = raw_logits.cpu().numpy()
            probs = torch.sigmoid(torch.tensor(logits)).numpy()
            preds = (probs >= 0.5).astype(int)

        return roc_auc(labels, probs), pr_auc(labels, probs), binary_accuracy(labels, preds), binary_f1(labels, preds), logits, labels, probs

    def on_train_epoch_end(self):
        roc_auc, pr_auc, acc, f1, _, _, _ = self.calculate_metrics(self.train_step_outputs)
        self.log("train_auc", roc_auc, prog_bar=True)
        self.log("train_aupr", pr_auc)
        self.log("train_acc", acc)
        self.log("train_f1", f1)
        self.train_step_outputs.clear()

    def on_validation_epoch_end(self):
        auc_value, aupr_value, acc, f1, logits, labels, probs = self.calculate_metrics(self.val_step_outputs)

        pos_mask = labels == 1
        neg_mask = labels == 0
        pos_logit_mean = logits[pos_mask].mean() if pos_mask.any() else 0.0
        neg_logit_mean = logits[neg_mask].mean() if neg_mask.any() else 0.0
        pos_prob_mean = probs[pos_mask].mean() if pos_mask.any() else 0.0
        neg_prob_mean = probs[neg_mask].mean() if neg_mask.any() else 0.0
        logit_std = logits.std() if logits.size else 0.0

        self.log("val_auc", auc_value, prog_bar=True)
        self.log("val_aupr", aupr_value, prog_bar=True)
        self.log("val_score", auc_value + aupr_value, prog_bar=True)
        self.log("val_average_precision", average_precision(labels, probs))
        self.log("val_brier", float(np.mean((probs - labels) ** 2)))
        self.log("val_ece", expected_calibration_error(labels, probs))
        self.log("val_acc", acc)
        self.log("val_f1", f1)
        self.log("val_pos_logit", float(pos_logit_mean))
        self.log("val_neg_logit", float(neg_logit_mean))
        self.log("val_pos_prob", float(pos_prob_mean))
        self.log("val_neg_prob", float(neg_prob_mean))
        self.log("val_logit_std", float(logit_std))

        if self.val_step_outputs and all("pathway_correction" in x for x in self.val_step_outputs):
            corrections = torch.cat([x["pathway_correction"] for x in self.val_step_outputs]).cpu().numpy()
            correction_auc = roc_auc(labels, corrections)
            correction_aupr = pr_auc(labels, corrections)
            self.log("val_correction_auc", correction_auc, prog_bar=False)
            self.log("val_correction_aupr", correction_aupr, prog_bar=False)
            self.log("val_correction_score", correction_auc + correction_aupr, prog_bar=True)

        if self.val_step_outputs and all("base_logits" in x for x in self.val_step_outputs):
            raw_base_logits = torch.cat([x["base_logits"] for x in self.val_step_outputs]).cpu()
            if raw_base_logits.ndim == 2 and raw_base_logits.size(1) == 2:
                base_probs = torch.softmax(raw_base_logits, dim=1)[:, 1].numpy()
            else:
                base_probs = torch.sigmoid(raw_base_logits.reshape(-1)).numpy()
            base_auc = roc_auc(labels, base_probs)
            base_aupr = pr_auc(labels, base_probs)
            self.log("val_base_auc", base_auc, prog_bar=False)
            self.log("val_base_aupr", base_aupr, prog_bar=False)
            self.log("val_base_score", base_auc + base_aupr, prog_bar=False)

        if self.val_step_outputs and all("metabolic_compatibility_logit" in x for x in self.val_step_outputs):
            compatibility = torch.cat(
                [x["metabolic_compatibility_logit"] for x in self.val_step_outputs]
            ).cpu().numpy()
            metabolic_auc = roc_auc(labels, compatibility)
            metabolic_aupr = pr_auc(labels, compatibility)
            self.log("val_metabolic_auc", metabolic_auc, prog_bar=False)
            self.log("val_metabolic_aupr", metabolic_aupr, prog_bar=False)
            self.log("val_metabolic_score", metabolic_auc + metabolic_aupr, prog_bar=True)

        self.val_step_outputs.clear()

    def _calc_and_log_metrics(self, outputs, prefix):
        if not outputs:
            return

        roc_auc, pr_auc, acc, f1, logits, labels, probs = self.calculate_metrics(outputs)
        raw_logits = torch.cat([x["logits"] for x in outputs]).detach().cpu()
        if raw_logits.ndim == 2 and raw_logits.size(1) == 2:
            preds = torch.argmax(raw_logits, dim=1).numpy().astype(int)
        else:
            preds = (probs >= 0.5).astype(int)

        self.log(f"{prefix}_auc", roc_auc, prog_bar=True)
        self.log(f"{prefix}_aupr", pr_auc, prog_bar=True)
        self.log(f"{prefix}_score", roc_auc + pr_auc)
        self.log(f"{prefix}_average_precision", average_precision(labels, probs))
        self.log(f"{prefix}_brier", float(np.mean((probs - labels) ** 2)))
        self.log(f"{prefix}_ece", expected_calibration_error(labels, probs))
        self.log(f"{prefix}_acc", acc)
        self.log(f"{prefix}_f1", f1)

        if prefix == "test" and self.config.get("evaluation", {}).get("export_predictions", True):
            self._export_test_predictions(outputs, logits, labels, probs, preds)

        outputs.clear()

    def on_test_epoch_end(self):
        self._calc_and_log_metrics(self.test_step_outputs, "test")

    def _export_test_predictions(self, outputs, logits, labels, probs, preds):
        run_dir = self.config.get("experiment", {}).get("run_dir")
        if not run_dir:
            return

        metas = []
        for batch_output in outputs:
            metas.extend(batch_output.get("metas") or [])

        rows = []
        diagnostic_keys = [
            "metabolic_compatibility_logit",
            "metabolic_relevance",
            "router_rescue_probability",
            "router_suppress_probability",
            "router_abstain_probability",
            "rescue_magnitude",
            "suppress_magnitude",
            "pathway_base_uncertainty",
            "pathway_correction",
        ]
        diagnostics = {}
        if outputs and all("base_logits" in batch_output for batch_output in outputs):
            raw_base_logits = torch.cat([batch_output["base_logits"] for batch_output in outputs]).cpu()
            if raw_base_logits.ndim == 2 and raw_base_logits.size(1) == 2:
                diagnostics["base_logit"] = (raw_base_logits[:, 1] - raw_base_logits[:, 0]).numpy()
                diagnostics["base_prob"] = torch.softmax(raw_base_logits, dim=1)[:, 1].numpy()
            else:
                base_logit = raw_base_logits.reshape(-1)
                diagnostics["base_logit"] = base_logit.numpy()
                diagnostics["base_prob"] = torch.sigmoid(base_logit).numpy()
        for key in diagnostic_keys:
            if outputs and all(key in batch_output for batch_output in outputs):
                diagnostics[key] = torch.cat([batch_output[key] for batch_output in outputs]).cpu().numpy()
        for i, label in enumerate(labels):
            meta = metas[i] if i < len(metas) and metas[i] is not None else {}
            row = {
                "row_index": meta.get("row_index", i),
                "head": meta.get("head", ""),
                "tail": meta.get("tail", ""),
                "label": int(label),
                "logit": float(logits[i]),
                "prob": float(probs[i]),
                "pred": int(preds[i]),
            }
            for key, values in diagnostics.items():
                row[key] = float(values[i])
            rows.append(row)

        output_path = os.path.join(run_dir, "test_predictions.csv")
        pd.DataFrame(rows).to_csv(output_path, index=False)

    def configure_optimizers(self):
        trainer_config = self.config["trainer"]
        global_lr = float(trainer_config["learning_rate"])
        weight_decay = float(trainer_config["weight_decay"])
        layer_rates = trainer_config.get("layer_rates", {})

        optimizer_name = trainer_config.get("optimizer", "AdamW")
        optimizer_cls = getattr(torch.optim, optimizer_name)

        if not layer_rates:
            trainable_params = [parameter for parameter in self.model.parameters() if parameter.requires_grad]
            if not trainable_params:
                raise ValueError("Model has no trainable parameters")
            optimizer = optimizer_cls(trainable_params, lr=global_lr, weight_decay=weight_decay)
        else:
            param_groups = []
            assigned_params = set()

            for layer_name, specific_lr in layer_rates.items():
                group_params = []
                for name, param in self.model.named_parameters():
                    if param.requires_grad and layer_name in name:
                        group_params.append(param)
                        assigned_params.add(name)

                if group_params:
                    param_groups.append({"params": group_params, "lr": float(specific_lr)})

            default_params = [
                parameter
                for name, parameter in self.model.named_parameters()
                if parameter.requires_grad and name not in assigned_params
            ]
            if default_params:
                param_groups.append({"params": default_params, "lr": global_lr})

            optimizer = optimizer_cls(param_groups, weight_decay=weight_decay)

        sched_config = trainer_config.get("scheduler", {})
        if not sched_config or not sched_config.get("active", False):
            return optimizer

        import torch.optim.lr_scheduler as lr_scheduler

        scheduler_class = getattr(lr_scheduler, sched_config["name"])
        scheduler = scheduler_class(optimizer, **sched_config.get("params", {}))
        scheduler_dict = {
            "scheduler": scheduler,
            "interval": sched_config.get("interval", "epoch"),
            "frequency": 1,
        }
        if scheduler.__class__.__name__ == "ReduceLROnPlateau":
            scheduler_dict["monitor"] = sched_config.get("monitor", trainer_config.get("monitor_metric", "val_score"))

        return [optimizer], [scheduler_dict]
