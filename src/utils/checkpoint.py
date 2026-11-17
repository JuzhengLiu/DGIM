"""Load a training checkpoint as an inference-only DGIM matcher."""

from copy import deepcopy

import torch

from src.dgim import DGIM, default_cfg, reparameter


def load_matcher(checkpoint_path, device='cuda', half=False):
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    # Original DGIM checkpoints have no architecture metadata and use RepVGG.
    config = deepcopy(checkpoint.get('dgim_config', default_cfg))
    state = {
        key.removeprefix('matcher.'): value
        for key, value in checkpoint['state_dict'].items()
        if key.startswith('matcher.')
    }
    matcher = DGIM(config)
    matcher.load_state_dict(state, strict=True)
    matcher.eval()
    matcher = reparameter(matcher).to(device)
    if half:
        matcher.half()
    return matcher
