import torch
import torch.nn as nn
import torch.nn.functional as F


class WeightedBCEWithLogitsLoss(nn.Module):
    expects_outputs_dict = True

    def __init__(self, pos_weight=None, smoothing=0.0):
        super().__init__()
        self.smoothing = smoothing
        if pos_weight is None:
            self.register_buffer("pos_weight_tensor", None)
        else:
            self.register_buffer("pos_weight_tensor", torch.tensor([float(pos_weight)]))

    def forward(self, outputs, targets):
        logits = outputs["logits"]
        weights = outputs.get("sample_weight")
        if weights is None:
            weights = torch.ones_like(targets, dtype=logits.dtype)
        else:
            weights = weights.to(device=logits.device, dtype=logits.dtype)

        targets = targets.to(dtype=logits.dtype)
        if self.smoothing:
            targets = targets * (1 - self.smoothing) + self.smoothing / 2

        pos_weight = self.pos_weight_tensor
        if pos_weight is not None:
            pos_weight = pos_weight.to(device=logits.device, dtype=logits.dtype)

        loss = F.binary_cross_entropy_with_logits(
            logits,
            targets,
            pos_weight=pos_weight,
            reduction="none",
        )
        weighted = loss * weights
        return weighted.sum() / weights.sum().clamp(min=1e-9)


class CrossEntropyFromLogitsLoss(nn.Module):
    expects_outputs_dict = True

    def __init__(self, class_weight=None, label_smoothing=0.0):
        super().__init__()
        self.label_smoothing = float(label_smoothing)
        if class_weight is None:
            self.register_buffer("class_weight_tensor", None)
        else:
            self.register_buffer("class_weight_tensor", torch.tensor(class_weight, dtype=torch.float32))

    def forward(self, outputs, targets):
        logits = outputs["logits"]
        weights = outputs.get("sample_weight")
        class_weight = self.class_weight_tensor
        if class_weight is not None:
            class_weight = class_weight.to(device=logits.device, dtype=logits.dtype)

        loss = F.cross_entropy(
            logits,
            targets.long(),
            weight=class_weight,
            reduction="none",
            label_smoothing=self.label_smoothing,
        )
        if weights is None:
            return loss.mean()

        weights = weights.to(device=logits.device, dtype=loss.dtype)
        return (loss * weights).sum() / weights.sum().clamp(min=1e-9)


class GatedResidualCrossEntropyLoss(nn.Module):
    expects_outputs_dict = True

    def __init__(
        self,
        class_weight=None,
        label_smoothing=0.0,
        gate_l1_weight=0.01,
        delta_l2_weight=0.0,
        correction_l1_weight=0.01,
        negative_positive_correction_weight=0.0,
        positive_negative_correction_weight=0.0,
        entropy_weight=0.0,
        distill_weight=0.0,
        distill_confidence=0.9,
    ):
        super().__init__()
        self.label_smoothing = float(label_smoothing)
        self.gate_l1_weight = float(gate_l1_weight)
        self.delta_l2_weight = float(delta_l2_weight)
        self.correction_l1_weight = float(correction_l1_weight)
        self.negative_positive_correction_weight = float(negative_positive_correction_weight)
        self.positive_negative_correction_weight = float(positive_negative_correction_weight)
        self.entropy_weight = float(entropy_weight)
        self.distill_weight = float(distill_weight)
        self.distill_confidence = float(distill_confidence)
        if class_weight is None:
            self.register_buffer("class_weight_tensor", None)
        else:
            self.register_buffer("class_weight_tensor", torch.tensor(class_weight, dtype=torch.float32))

    def _weighted_ce(self, logits, targets, weights):
        class_weight = self.class_weight_tensor
        if class_weight is not None:
            class_weight = class_weight.to(device=logits.device, dtype=logits.dtype)

        loss = F.cross_entropy(
            logits,
            targets.long(),
            weight=class_weight,
            reduction="none",
            label_smoothing=self.label_smoothing,
        )
        if weights is None:
            return loss.mean()
        weights = weights.to(device=logits.device, dtype=loss.dtype)
        return (loss * weights).sum() / weights.sum().clamp(min=1e-9)

    def _base_distill_loss(self, logits, base_logits):
        if base_logits is None or self.distill_weight <= 0:
            return logits.sum() * 0.0

        with torch.no_grad():
            base_probs = torch.softmax(base_logits, dim=1)
            confidence = base_probs.max(dim=1).values
            keep = confidence >= self.distill_confidence
        if not keep.any():
            return logits.sum() * 0.0

        return F.kl_div(
            F.log_softmax(logits[keep], dim=1),
            base_probs[keep],
            reduction="batchmean",
        )

    def forward(self, outputs, targets):
        logits = outputs["logits"]
        weights = outputs.get("sample_weight")
        loss = self._weighted_ce(logits, targets, weights)

        gate = outputs.get("pathway_gate")
        if gate is not None and self.gate_l1_weight > 0:
            loss = loss + self.gate_l1_weight * gate.mean()

        delta = outputs.get("pathway_delta")
        if delta is not None and self.delta_l2_weight > 0:
            loss = loss + self.delta_l2_weight * (delta ** 2).mean()

        correction = outputs.get("pathway_correction")
        if correction is not None and self.correction_l1_weight > 0:
            loss = loss + self.correction_l1_weight * correction.abs().mean()
        if correction is not None and self.negative_positive_correction_weight > 0:
            neg_mask = targets.to(device=correction.device) < 0.5
            if neg_mask.any():
                loss = loss + self.negative_positive_correction_weight * F.relu(correction[neg_mask]).mean()
        if correction is not None and self.positive_negative_correction_weight > 0:
            pos_mask = targets.to(device=correction.device) > 0.5
            if pos_mask.any():
                loss = loss + self.positive_negative_correction_weight * F.relu(-correction[pos_mask]).mean()

        entropy = outputs.get("pathway_contact_entropy")
        if entropy is not None and self.entropy_weight > 0:
            if gate is not None:
                loss = loss + self.entropy_weight * (entropy * gate.detach()).mean()
            else:
                loss = loss + self.entropy_weight * entropy.mean()

        base_logits = outputs.get("base_logits")
        if base_logits is not None and self.distill_weight > 0:
            loss = loss + self.distill_weight * self._base_distill_loss(logits, base_logits.detach())

        return loss


class PathwayResidualPULoss(nn.Module):
    expects_outputs_dict = True

    def __init__(
        self,
        pairwise_weight=1.0,
        pairwise_margin=0.25,
        pos_margin_weight=0.1,
        pos_margin=0.1,
        unlabeled_positive_penalty_weight=0.02,
        unlabeled_margin=0.25,
        center_weight=0.1,
        gate_l1_weight=0.005,
        correction_l1_weight=0.005,
        delta_l2_weight=0.0001,
        entropy_weight=0.0,
    ):
        super().__init__()
        self.pairwise_weight = float(pairwise_weight)
        self.pairwise_margin = float(pairwise_margin)
        self.pos_margin_weight = float(pos_margin_weight)
        self.pos_margin = float(pos_margin)
        self.unlabeled_positive_penalty_weight = float(unlabeled_positive_penalty_weight)
        self.unlabeled_margin = float(unlabeled_margin)
        self.center_weight = float(center_weight)
        self.gate_l1_weight = float(gate_l1_weight)
        self.correction_l1_weight = float(correction_l1_weight)
        self.delta_l2_weight = float(delta_l2_weight)
        self.entropy_weight = float(entropy_weight)

    def forward(self, outputs, targets):
        correction = outputs["pathway_correction"]
        labels = targets.to(device=correction.device, dtype=torch.float32)
        pos_mask = labels > 0.5
        neg_mask = ~pos_mask
        loss = correction.sum() * 0.0

        if pos_mask.any() and neg_mask.any() and self.pairwise_weight > 0:
            pos_scores = correction[pos_mask]
            neg_scores = correction[neg_mask]
            pairwise = F.softplus(self.pairwise_margin - (pos_scores[:, None] - neg_scores[None, :])).mean()
            loss = loss + self.pairwise_weight * pairwise

        if pos_mask.any() and self.pos_margin_weight > 0:
            loss = loss + self.pos_margin_weight * F.softplus(self.pos_margin - correction[pos_mask]).mean()

        if neg_mask.any() and self.unlabeled_positive_penalty_weight > 0:
            neg_scores = correction[neg_mask]
            loss = loss + self.unlabeled_positive_penalty_weight * F.softplus(
                neg_scores - self.unlabeled_margin
            ).mean()

        if self.center_weight > 0:
            loss = loss + self.center_weight * correction.mean().pow(2)

        gate = outputs.get("pathway_gate")
        if gate is not None and self.gate_l1_weight > 0:
            loss = loss + self.gate_l1_weight * gate.mean()

        if self.correction_l1_weight > 0:
            loss = loss + self.correction_l1_weight * correction.abs().mean()

        delta = outputs.get("pathway_delta")
        if delta is not None and self.delta_l2_weight > 0:
            loss = loss + self.delta_l2_weight * delta.pow(2).mean()

        entropy = outputs.get("pathway_contact_entropy")
        if entropy is not None and self.entropy_weight > 0:
            if gate is not None:
                loss = loss + self.entropy_weight * (entropy * gate.detach()).mean()
            else:
                loss = loss + self.entropy_weight * entropy.mean()

        return loss


class RoutedMetabolicPretrainingLoss(nn.Module):
    """PU-aware pretraining for the V3.7 metabolic compatibility encoder."""

    expects_outputs_dict = True

    def __init__(
        self,
        pairwise_weight=1.0,
        pairwise_margin=0.2,
        positive_margin_weight=0.1,
        positive_margin=0.1,
        unlabeled_penalty_weight=0.03,
        unlabeled_margin=0.25,
        relevance_weight=0.1,
        unlabeled_relevance_weight=0.25,
        center_weight=0.05,
    ):
        super().__init__()
        self.pairwise_weight = float(pairwise_weight)
        self.pairwise_margin = float(pairwise_margin)
        self.positive_margin_weight = float(positive_margin_weight)
        self.positive_margin = float(positive_margin)
        self.unlabeled_penalty_weight = float(unlabeled_penalty_weight)
        self.unlabeled_margin = float(unlabeled_margin)
        self.relevance_weight = float(relevance_weight)
        self.unlabeled_relevance_weight = float(unlabeled_relevance_weight)
        self.center_weight = float(center_weight)

    @staticmethod
    def _weighted_mean(values, weights):
        if weights is None:
            return values.mean()
        weights = weights.to(device=values.device, dtype=values.dtype)
        return (values * weights).sum() / weights.sum().clamp(min=1e-9)

    def forward(self, outputs, targets):
        score = outputs["metabolic_compatibility_logit"]
        relevance = outputs["metabolic_relevance"]
        labels = targets.to(device=score.device, dtype=score.dtype)
        weights = outputs.get("sample_weight")
        pos_mask = labels > 0.5
        unlabeled_mask = ~pos_mask
        loss = score.sum() * 0.0

        if pos_mask.any() and unlabeled_mask.any() and self.pairwise_weight > 0:
            pairwise = F.softplus(
                self.pairwise_margin - (score[pos_mask, None] - score[None, unlabeled_mask])
            )
            loss = loss + self.pairwise_weight * pairwise.mean()

        if pos_mask.any() and self.positive_margin_weight > 0:
            pos_loss = F.softplus(self.positive_margin - score[pos_mask])
            pos_weights = weights[pos_mask] if weights is not None else None
            loss = loss + self.positive_margin_weight * self._weighted_mean(pos_loss, pos_weights)

        if unlabeled_mask.any() and self.unlabeled_penalty_weight > 0:
            unlabeled_loss = F.softplus(score[unlabeled_mask] - self.unlabeled_margin)
            unlabeled_weights = weights[unlabeled_mask] if weights is not None else None
            loss = loss + self.unlabeled_penalty_weight * self._weighted_mean(
                unlabeled_loss,
                unlabeled_weights,
            )

        if self.relevance_weight > 0:
            relevance_weights = torch.where(
                pos_mask,
                torch.ones_like(labels),
                torch.full_like(labels, self.unlabeled_relevance_weight),
            )
            if weights is not None:
                relevance_weights = relevance_weights * weights.to(relevance_weights)
            relevance_loss = F.binary_cross_entropy(
                relevance.clamp(min=1e-6, max=1.0 - 1e-6),
                labels,
                reduction="none",
            )
            loss = loss + self.relevance_weight * self._weighted_mean(
                relevance_loss,
                relevance_weights,
            )

        if self.center_weight > 0:
            loss = loss + self.center_weight * score.mean().square()
        return loss


class RoutedResidualMultiTaskLoss(nn.Module):
    """Downstream error-aware loss for rescue/suppress/abstain routing."""

    expects_outputs_dict = True

    def __init__(
        self,
        class_weight=None,
        label_smoothing=0.0,
        router_weight=0.25,
        router_gamma=2.0,
        distill_weight=0.25,
        correction_l1_weight=0.01,
        activity_weight=0.01,
        brier_weight=0.05,
        require_oof_targets=True,
    ):
        super().__init__()
        self.label_smoothing = float(label_smoothing)
        self.router_weight = float(router_weight)
        self.router_gamma = float(router_gamma)
        self.distill_weight = float(distill_weight)
        self.correction_l1_weight = float(correction_l1_weight)
        self.activity_weight = float(activity_weight)
        self.brier_weight = float(brier_weight)
        self.require_oof_targets = bool(require_oof_targets)
        if class_weight is None:
            self.register_buffer("class_weight_tensor", None)
        else:
            self.register_buffer("class_weight_tensor", torch.tensor(class_weight, dtype=torch.float32))

    @staticmethod
    def _positive_probability(logits):
        if logits.ndim == 2 and logits.size(1) == 2:
            return torch.softmax(logits, dim=1)[:, 1]
        return torch.sigmoid(logits.reshape(-1))

    @staticmethod
    def _weighted_mean(values, weights):
        if weights is None:
            return values.mean()
        weights = weights.to(device=values.device, dtype=values.dtype)
        return (values * weights).sum() / weights.sum().clamp(min=1e-9)

    def _classification_loss(self, logits, labels, weights):
        class_weight = self.class_weight_tensor
        if class_weight is not None:
            class_weight = class_weight.to(device=logits.device, dtype=logits.dtype)
        if logits.ndim == 2 and logits.size(1) == 2:
            losses = F.cross_entropy(
                logits,
                labels.long(),
                weight=class_weight,
                reduction="none",
                label_smoothing=self.label_smoothing,
            )
        else:
            smooth_labels = labels
            if self.label_smoothing:
                smooth_labels = labels * (1.0 - self.label_smoothing) + 0.5 * self.label_smoothing
            losses = F.binary_cross_entropy_with_logits(
                logits.reshape(-1),
                smooth_labels,
                reduction="none",
            )
        return self._weighted_mean(losses, weights)

    def _teacher_probability(self, outputs):
        oof_probability = outputs.get("oof_base_prob")
        if oof_probability is not None:
            return oof_probability.detach().clamp(min=1e-5, max=1.0 - 1e-5)
        if self.require_oof_targets and self.training:
            raise ValueError(
                "RoutedResidualMultiTaskLoss requires oof_base_prob metadata. "
                "Build cross-fitted router targets or set require_oof_targets=false only for a declared diagnostic run."
            )
        return self._positive_probability(outputs["base_logits"]).detach()

    def forward(self, outputs, targets):
        logits = outputs["logits"]
        labels = targets.to(device=logits.device, dtype=logits.dtype)
        weights = outputs.get("sample_weight")
        final_probability = self._positive_probability(logits)
        live_base_probability = self._positive_probability(outputs["base_logits"]).detach()
        teacher_probability = self._teacher_probability(outputs)

        loss = self._classification_loss(logits, labels, weights)

        router_logits = outputs["router_logits"]
        rescue_weight = labels * (1.0 - teacher_probability).pow(self.router_gamma)
        suppress_weight = (1.0 - labels) * teacher_probability.pow(self.router_gamma)
        abstain_weight = (1.0 - rescue_weight - suppress_weight).clamp(min=0.0)
        router_targets = torch.stack([rescue_weight, suppress_weight, abstain_weight], dim=1)
        router_targets = router_targets / router_targets.sum(dim=1, keepdim=True).clamp(min=1e-9)
        router_loss = -(router_targets * F.log_softmax(router_logits, dim=1)).sum(dim=1)
        loss = loss + self.router_weight * self._weighted_mean(router_loss, weights)

        relevance = outputs["metabolic_relevance"].detach()
        base_confidence = (2.0 * (live_base_probability - 0.5).abs()).clamp(min=0.0, max=1.0)
        distill_weights = base_confidence * (1.0 - relevance)
        base_distribution = torch.stack([1.0 - live_base_probability, live_base_probability], dim=1)
        final_distribution = torch.stack([1.0 - final_probability, final_probability], dim=1)
        distill = (
            base_distribution
            * (torch.log(base_distribution.clamp(min=1e-7)) - torch.log(final_distribution.clamp(min=1e-7)))
        ).sum(dim=1)
        loss = loss + self.distill_weight * self._weighted_mean(distill * distill_weights, weights)

        correction = outputs["pathway_correction"]
        if self.correction_l1_weight > 0:
            loss = loss + self.correction_l1_weight * correction.abs().mean()
        if self.activity_weight > 0:
            loss = loss + self.activity_weight * (1.0 - outputs["router_abstain_probability"]).mean()
        if self.brier_weight > 0:
            brier = (final_probability - labels).square()
            loss = loss + self.brier_weight * self._weighted_mean(brier, weights)
        return loss


class PositiveOnlyContrastiveLoss(nn.Module):
    expects_outputs_dict = True

    def __init__(
        self,
        contrastive_weight=1.0,
        positive_logit_weight=0.05,
        temperature=0.1,
        margin=1.0,
    ):
        super().__init__()
        self.contrastive_weight = float(contrastive_weight)
        self.positive_logit_weight = float(positive_logit_weight)
        self.temperature = float(temperature)
        self.margin = float(margin)

    def _positive_logit_margin(self, logits, labels):
        pos_mask = labels > 0.5
        if not pos_mask.any():
            return logits.sum() * 0.0

        if logits.ndim == 2 and logits.size(1) == 2:
            score = logits[:, 1] - logits[:, 0]
        else:
            score = logits
        return F.softplus(self.margin - score[pos_mask]).mean()

    def _contrastive(self, prot_pool, lig_pool, labels):
        pos_mask = labels > 0.5
        if pos_mask.sum() <= 1:
            return prot_pool.sum() * 0.0

        prot_norm = F.normalize(prot_pool, dim=-1)
        lig_norm = F.normalize(lig_pool, dim=-1)
        sim = torch.matmul(prot_norm, lig_norm.T) / self.temperature
        targets = torch.arange(sim.size(0), device=sim.device)

        loss_p2l = F.cross_entropy(sim[pos_mask], targets[pos_mask])
        loss_l2p = F.cross_entropy(sim.T[pos_mask], targets[pos_mask])
        return 0.5 * (loss_p2l + loss_l2p)

    def forward(self, outputs, targets):
        labels = targets.to(dtype=torch.float32, device=outputs["logits"].device)
        contrastive = self._contrastive(outputs["prot_pool"], outputs["lig_pool"], labels)
        pos_margin = self._positive_logit_margin(outputs["logits"], labels)
        return self.contrastive_weight * contrastive + self.positive_logit_weight * pos_margin


class ContactMapCELoss(nn.Module):
    expects_outputs_dict = True

    def __init__(
        self,
        class_weight=None,
        label_smoothing=0.0,
        contact_aux_weight=0.05,
        positive_margin=0.5,
        negative_margin=-0.5,
        positive_entropy_weight=0.01,
    ):
        super().__init__()
        self.label_smoothing = float(label_smoothing)
        self.contact_aux_weight = float(contact_aux_weight)
        self.positive_margin = float(positive_margin)
        self.negative_margin = float(negative_margin)
        self.positive_entropy_weight = float(positive_entropy_weight)
        if class_weight is None:
            self.register_buffer("class_weight_tensor", None)
        else:
            self.register_buffer("class_weight_tensor", torch.tensor(class_weight, dtype=torch.float32))

    def forward(self, outputs, targets):
        logits = outputs["logits"]
        weights = outputs.get("sample_weight")
        class_weight = self.class_weight_tensor
        if class_weight is not None:
            class_weight = class_weight.to(device=logits.device, dtype=logits.dtype)

        ce = F.cross_entropy(
            logits,
            targets.long(),
            weight=class_weight,
            reduction="none",
            label_smoothing=self.label_smoothing,
        )
        if weights is not None:
            weights = weights.to(device=logits.device, dtype=ce.dtype)
            ce_loss = (ce * weights).sum() / weights.sum().clamp(min=1e-9)
        else:
            ce_loss = ce.mean()

        contact_logit = outputs.get("contact_bag_logit")
        if contact_logit is None or self.contact_aux_weight <= 0:
            return ce_loss

        labels = targets.to(device=contact_logit.device, dtype=contact_logit.dtype)
        pos_mask = labels > 0.5
        neg_mask = ~pos_mask

        aux_terms = []
        if pos_mask.any():
            aux_terms.append(F.softplus(self.positive_margin - contact_logit[pos_mask]).mean())
            entropy = outputs.get("contact_entropy")
            if entropy is not None and self.positive_entropy_weight > 0:
                entropy = entropy.to(device=contact_logit.device, dtype=contact_logit.dtype)
                aux_terms.append(self.positive_entropy_weight * entropy[pos_mask].mean())
        if neg_mask.any():
            aux_terms.append(F.softplus(contact_logit[neg_mask] - self.negative_margin).mean())

        if not aux_terms:
            return ce_loss
        return ce_loss + self.contact_aux_weight * sum(aux_terms)


class MILFocalLoss(nn.Module):
    def __init__(self, alpha=0.6, gamma=2.0, label_smoothing=0.1, reduction='mean'):
        super(MILFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.label_smoothing = label_smoothing
        self.reduction = reduction

    def forward(self, inputs, targets):
        # Apply label smoothing
        smoothed_targets = targets * (1.0 - self.label_smoothing) + 0.5 * self.label_smoothing
        
        # 1. Compute BCE with smoothed targets
        bce_loss = F.binary_cross_entropy_with_logits(inputs, smoothed_targets, reduction='none')
        
        # 2. Get the probabilities for the focusing mechanism (using original inputs)
        pt = torch.exp(-F.binary_cross_entropy_with_logits(inputs, targets, reduction='none'))
        
        # 3. Calculate the Focal weighting
        focal_weight = (1 - pt) ** self.gamma
        
        # 4. Apply the alpha class weighting
        alpha_weight = torch.where(targets == 1, self.alpha, 1 - self.alpha)
        
        # 5. Final Loss
        loss = alpha_weight * focal_weight * bce_loss
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss
        

class BootstrappedFocalLoss(nn.Module):
    def __init__(self, trust_label=0.8, alpha=0.25, gamma=2.0, reduction='mean'):
        """
        trust_label (beta): 0.8 means 80% database, 20% model prediction.
        alpha: Weighting factor for the positive class (BindingDB is highly biased).
        gamma: Focusing parameter to down-weight easy examples.
        """
        super().__init__()
        self.trust = trust_label
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, targets):
        """
        logits: Unnormalized network outputs (Batch_Size,)
        targets: Ground truth binary labels (Batch_Size,)
        """
        # 1. Detach the probabilities for target bootstrapping to prevent 
        # the model from taking a shortcut by just moving the target to 0 loss.
        with torch.no_grad():
            detached_probs = torch.sigmoid(logits)
            # Create the bootstrapped continuous targets
            soft_targets = (self.trust * targets) + ((1 - self.trust) * detached_probs)

        # 2. Get active probabilities and stable log probabilities
        probs = torch.sigmoid(logits)
        log_p = F.logsigmoid(logits)
        log_inv_p = F.logsigmoid(-logits) # Numerically stable log(1-p)

        # 3. Calculate Continuous Alpha
        alpha_t = soft_targets * self.alpha + (1 - soft_targets) * (1 - self.alpha)

        # 4. Calculate Distance-Based Focal Weight
        # How far is the prediction from the newly accepted 'truth'?
        focal_weight = torch.abs(soft_targets - probs) ** self.gamma

        # 5. Calculate Soft Binary Cross Entropy
        bce_pos = soft_targets * log_p
        bce_neg = (1 - soft_targets) * log_inv_p

        # 6. Assemble the final loss
        loss = -alpha_t * focal_weight * (bce_pos + bce_neg)

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss

class BootstrappedBCE(nn.Module):
    def __init__(self, trust_label=0.8):
        """
        trust_label (beta): How much to trust the dataset vs the model.
        0.8 means 80% dataset label, 20% model's prediction.
        """
        super().__init__()
        self.trust = trust_label
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, logits, targets):
        # 1. Get the model's current probabilities (detached so we don't double-backprop)
        with torch.no_grad():
            probs = torch.sigmoid(logits)
        
        # 2. Blend the hard target with the model's confidence
        # If target=0, but prob=0.9, new target becomes 0.18
        soft_targets = (self.trust * targets) + ((1 - self.trust) * probs)
        
        # 3. Calculate BCE against the new softened targets
        return self.bce(logits, soft_targets)



class BootstrappedBCEWithLogitPenalty(nn.Module):
    def __init__(self, trust_label=0.85, logit_penalty_weight=0.05):
        """
        logit_penalty_weight: Controls how strictly we punish large logits.
        """
        super().__init__()
        self.trust = trust_label
        self.penalty_weight = logit_penalty_weight
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, logits, targets):
        # 1. Bootstrapped Soft Targets
        with torch.no_grad():
            probs = torch.sigmoid(logits)
        soft_targets = (self.trust * targets) + ((1 - self.trust) * probs)
        
        # 2. Standard BCE on Soft Targets
        bce_loss = self.bce(logits, soft_targets)
        
        # 3. THE STABILIZER: L2 Penalty on the Logits
        logit_penalty = self.penalty_weight * torch.mean(logits ** 2)
        
        return bce_loss + logit_penalty
    
class AsymmetricBootstrappedBCE(nn.Module):
    def __init__(self, trust_label=0.95, pos_penalty_weight=0.00, neg_penalty_weight=0.05, pos_weight=None):
        """
        Added pos_weight to handle the BindingDB 60:40 imbalance directly in the BCE calculation.
        """
        super().__init__()
        self.trust = trust_label
        self.pos_penalty_weight = pos_penalty_weight
        self.neg_penalty_weight = neg_penalty_weight
        self.bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    def forward(self, logits, targets):
        # 1. Bootstrapped Soft Targets (Prevents over-confidence on noisy KEGG labels)
        with torch.no_grad():
            probs = torch.sigmoid(logits)
        soft_targets = (self.trust * targets) + ((1 - self.trust) * probs)
        
        # 2. Standard BCE on Soft Targets
        bce_loss = self.bce(logits, soft_targets)
        
        # 3. Asymmetric L2 Penalty
        # Masks to separate positive and negative ground-truth instances
        pos_mask = (targets > 0.5).float()
        neg_mask = (targets <= 0.5).float()
        
        # Apply penalties selectively. 
        # Squaring the masked logits ensures we only penalize the magnitude of the active class.
        pos_penalty = self.pos_penalty_weight * torch.mean((logits * pos_mask) ** 2)
        neg_penalty = self.neg_penalty_weight * torch.mean((logits * neg_mask) ** 2)
        
        return bce_loss + pos_penalty + neg_penalty
    


class DPIContrastiveLoss(nn.Module):
    """
    Combined loss for SpanGuidedDPI Phase 1.

    Two components:

    1. BCE loss  — primary supervision signal.
       Receives the scalar logit per pair and the binary binding label.
       pos_weight upweights the positive class to counter class imbalance.

    2. Contrastive alignment loss (InfoNCE-style) — representation alignment.
       Forces the protein and ligand 128-dim projection spaces to align:
       binding pairs (label=1) should have high cosine similarity,
       non-binding pairs (label=0) should have low cosine similarity.

       Without this loss, the CNN protein subspace and the ligand projector
       subspace are geometrically unrelated. Cross-attention dot products
       between unaligned spaces are random with respect to binding affinity,
       which is the root cause of the low positive logit problem.

       The contrastive loss is computed as follows:
         - Normalise prot_pool and lig_pool to unit vectors.
         - Build a (B, B) cosine similarity matrix scaled by temperature.
         - For each anchor protein, treat its paired ligand as the positive
           and all other ligands in the batch as negatives (in-batch negatives).
           Do the same in the ligand-to-protein direction.
         - Average the two cross-entropy losses.

       Note: this is only meaningful for binding pairs (label=1). Non-binding
       pairs are valid negatives but should not be treated as anchors, because
       we do not want to push non-binding protein/ligand representations apart —
       we only want to pull binding pairs together. The mask below handles this.

    Args:
        pos_weight (float): Weight for the positive class in BCE loss.
                            For a 74:26 imbalance, use 74/26 ≈ 2.85.
                            For a 50:50 dataset, use 1.0.
        contrastive_weight (float): Scalar multiplier for the contrastive loss.
                            Start at 0.1. If val_pos_logit does not rise after
                            5 epochs, increase to 0.3. Do not exceed 1.0 as it
                            will dominate the primary task.
        temperature (float): Softmax temperature for the contrastive similarity
                            matrix. Lower = sharper distribution, harder negatives.
                            0.07 is standard (MoCo / SimCLR default).
    """

    expects_outputs_dict = True

    def __init__(
        self,
        pos_weight: float = 1.0,
        contrastive_weight: float = 0.1,
        temperature: float = 0.07,
        smoothing: float = 0.1
    ):
        super().__init__()
        self.contrastive_weight = contrastive_weight
        self.temperature = temperature
        self.smoothing = smoothing

        # Register as buffer so it moves to the correct device automatically
        # when loss_fn.to(device) is called (or via Lightning's auto device placement)
        self.register_buffer(
            'pos_weight_tensor',
            torch.tensor([pos_weight])
        )

    def bce_loss(self, logits, labels):
        # Smooth labels: 0 → 0.1, 1 → 0.9
        smooth_labels = labels.float() * (1 - self.smoothing) + self.smoothing / 2
        return F.binary_cross_entropy_with_logits(
            logits,
            smooth_labels,
            pos_weight=self.pos_weight_tensor
        )

    def contrastive_loss(
        self,
        prot_pool: torch.Tensor,    # (B, 128)
        lig_pool: torch.Tensor,     # (B, 128)
        labels: torch.Tensor        # (B,)  binary 0/1
    ) -> torch.Tensor:
        """
        In-batch InfoNCE contrastive alignment loss.

        Only binding pairs (label=1) serve as anchors. This is deliberate:
        we want to pull together protein-ligand representations for pairs
        that actually interact. Non-binding pairs serve as implicit negatives
        through the in-batch construction of the similarity matrix.

        If the batch contains no positive pairs (edge case with small batches
        and severe imbalance), returns 0.0 and contributes no gradient.
        """
        pos_mask = labels.bool()                                # (B,)

        # If no positive pairs in this batch, skip contrastive loss entirely.
        # This can happen with small batch sizes on imbalanced datasets.
        if pos_mask.sum() == 0:
            return torch.tensor(0.0, device=prot_pool.device, requires_grad=False)

        # L2-normalise both pools to the unit hypersphere.
        # Cosine similarity then reduces to a dot product, which is efficient.
        prot_norm = F.normalize(prot_pool, dim=-1)              # (B, 128)
        lig_norm  = F.normalize(lig_pool,  dim=-1)              # (B, 128)

        # Full (B, B) cosine similarity matrix, scaled by temperature.
        # sim[i, j] = cosine_similarity(protein_i, ligand_j) / temperature
        # The diagonal holds the similarity of each pair with itself (the
        # true positive for that anchor).
        sim_matrix = torch.matmul(prot_norm, lig_norm.T) / self.temperature  # (B, B)

        # Ground-truth targets for cross-entropy: for anchor i, the positive
        # target is position i (its own paired ligand).
        # Shape: (B,) — integer index of the positive for each row.
        targets = torch.arange(prot_pool.size(0), device=prot_pool.device)  # (B,)

        # Direction 1: protein anchors → find their paired ligand
        # Only compute for rows where label=1 (actual binding pairs).
        loss_p2l = F.cross_entropy(
            sim_matrix[pos_mask],       # (n_pos, B)
            targets[pos_mask]           # (n_pos,)
        )

        # Direction 2: ligand anchors → find their paired protein
        # Transpose the similarity matrix for this direction.
        loss_l2p = F.cross_entropy(
            sim_matrix.T[pos_mask],     # (n_pos, B)
            targets[pos_mask]           # (n_pos,)
        )

        return (loss_p2l + loss_l2p) / 2.0

    def forward(
        self,
        outputs: dict,
        labels: torch.Tensor        # (B,)
    ) -> torch.Tensor:
        """
        Args:
            outputs: the full dict returned by SpanGuidedDPI.forward().
                     Must contain keys: 'logits', 'prot_pool', 'lig_pool'.
            labels:  binary binding labels, shape (B,).

        Returns:
            Scalar total loss.
        """
        logits    = outputs['logits']       # (B,)
        prot_pool = outputs['prot_pool']    # (B, 128)
        lig_pool  = outputs['lig_pool']     # (B, 128)

        loss_bce         = self.bce_loss(logits, labels)
        loss_contrastive = self.contrastive_loss(prot_pool, lig_pool, labels)

        total_loss = loss_bce + self.contrastive_weight * loss_contrastive

        return total_loss
