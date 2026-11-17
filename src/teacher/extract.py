"""Frozen local DINOv3 teacher, used only by offline cache preparation."""

import importlib.util
from pathlib import Path

import torch
import torch.nn.functional as F

from src.teacher.cache import STRIDE


def load_frozen_teacher(source_dir, checkpoint_path, device):
    source = Path(source_dir) / 'convnext.py'
    spec = importlib.util.spec_from_file_location('_dgim_dinov3_convnext', source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    model = module.convnext_large()
    # Official backbone state_dict only: do not guess wrappers or discard keys.
    weights = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    model.load_state_dict(weights, strict=True)
    del weights
    model.requires_grad_(False)
    model.eval()
    return model.to(device)


def align_to_student_grid(features, image_mask):
    """Map native centers (8*i+3.5) onto student centers (8*i).

    Bilinear sampling uses align_corners=False. A target is valid only if all
    interpolation support lies in native stride-8 cells containing real image
    pixels, rather than padding. This excludes the first row/column as well as
    the padded boundary; validity is not a full receptive-field erosion.
    """
    batch, _, height, width = features.shape
    y = torch.arange(height, device=features.device, dtype=torch.float32) - 3.5 / STRIDE
    x = torch.arange(width, device=features.device, dtype=torch.float32) - 3.5 / STRIDE
    yy, xx = torch.meshgrid(y, x, indexing='ij')
    grid = torch.stack((2 * (xx + 0.5) / width - 1,
                        2 * (yy + 0.5) / height - 1), dim=-1)
    grid = grid.unsqueeze(0).expand(batch, -1, -1, -1)
    aligned = F.grid_sample(features.float(), grid, mode='bilinear',
                            padding_mode='zeros', align_corners=False)
    native_valid = F.avg_pool2d(image_mask.float(), kernel_size=STRIDE, stride=STRIDE)
    native_valid = (native_valid >= 1.0).float()
    support = F.grid_sample(native_valid, grid, mode='bilinear',
                            padding_mode='zeros', align_corners=False)
    inside = ((xx >= 0) & (xx <= width - 1) & (yy >= 0) & (yy <= height - 1))
    valid = (support[:, 0] >= 1 - 1e-6) & inside.unsqueeze(0)
    aligned = aligned.masked_fill(~valid[:, None], 0)
    return aligned, valid


@torch.inference_mode()
def extract_teacher_feature(model, image, mask, device):
    """Encode a clean [1,H,W] gray tensor using the official RGB normalization."""
    if image.shape[0] != 1 or any(side % STRIDE for side in image.shape[-2:]):
        raise ValueError('Teacher cache inputs must be grayscale with dimensions divisible by 8.')
    gray = image.unsqueeze(0).to(device)
    mean = gray.new_tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = gray.new_tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    normalized = (gray.repeat(1, 3, 1, 1) - mean) / std
    features = model(normalized)
    if mask is None:
        mask = torch.ones(image.shape[-2:], dtype=torch.bool)
    image_mask = mask[None, None].to(device)
    aligned, valid = align_to_student_grid(features, image_mask)
    return aligned[0], valid[0]
