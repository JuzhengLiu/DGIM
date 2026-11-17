"""Feature-only distillation; training never loads the frozen teacher."""

import torch
import torch.nn.functional as F


def teacher_weight(epoch, *, enabled, weight, ramp_epochs, stop_epoch):
    """Ramp over epochs 0..4, decay over 5..9, and disable at epoch 10.

    With the default epoch counts the multipliers are .2, .4, .6, .8, 1,
    1, .8, .6, .4, .2, followed by zero.
    """
    if not enabled or epoch >= stop_epoch:
        return 0.0
    if weight < 0 or not 0 < ramp_epochs < stop_epoch:
        raise ValueError("Teacher scheduling requires weight >= 0 and 0 < ramp_epochs < stop_epoch.")
    if epoch < ramp_epochs:
        multiplier = (epoch + 1) / ramp_epochs
    else:
        multiplier = (stop_epoch - epoch) / (stop_epoch - ramp_epochs)
    return float(weight) * multiplier


def masked_cosine_feature_loss(student, teacher, valid_mask, eps=1e-6):
    """Channel-normalized cosine distance on the aligned valid BCHW grid."""
    if student.ndim != 4 or teacher.shape != student.shape:
        raise ValueError(
            "Student projection and cached teacher must have identical BCHW shapes; "
            f"got {tuple(student.shape)} and {tuple(teacher.shape)}."
        )
    if valid_mask.shape != (student.shape[0], *student.shape[-2:]):
        raise ValueError("The cached teacher mask must have shape [B, H, W].")
    valid_mask = valid_mask.bool()
    target = teacher.detach().float().masked_fill(~valid_mask[:, None], 0.0)
    valid_mask = valid_mask & (torch.linalg.vector_norm(target, dim=1) > eps)
    prediction = F.normalize(student.float(), p=2, dim=1, eps=eps)
    target = F.normalize(target, p=2, dim=1, eps=eps)
    distance = 1.0 - (prediction * target).sum(dim=1).clamp(-1.0, 1.0)
    distance = distance.masked_fill(~valid_mask, 0.0)
    return distance.sum() / valid_mask.sum().clamp_min(1)
