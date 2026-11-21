import torch


def masked_mean(sequence, padding_mask=None, dim=1, eps=1e-9):
    if padding_mask is None:
        return sequence.mean(dim=dim)

    valid = (~padding_mask).to(sequence.dtype).unsqueeze(-1)
    return (sequence * valid).sum(dim=dim) / valid.sum(dim=dim).clamp(min=eps)


def expand_repeated_mask(padding_mask, repeats):
    if padding_mask is None:
        return None
    return padding_mask.repeat(1, repeats)


def zero_masked(sequence, padding_mask):
    if padding_mask is None:
        return sequence
    return sequence.masked_fill(padding_mask.unsqueeze(-1), 0.0)


def masked_softmax(scores, padding_mask=None, dim=-1):
    if padding_mask is None:
        return torch.softmax(scores, dim=dim)
    return torch.softmax(scores.masked_fill(padding_mask, float("-inf")), dim=dim)
