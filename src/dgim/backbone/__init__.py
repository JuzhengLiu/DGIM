from .backbone import RepVGG_8_2_align
from .repvit import RepViT_8_2


def build_backbone(config):
    if config['align_corner'] is not False:
        raise ValueError(f"DGIM.ALIGN_CORNER {config['align_corner']} not supported.")
    if tuple(config['resolution']) != (8, 2):
        raise ValueError(f"DGIM.RESOLUTION {config['resolution']} not supported.")
    if config['backbone_type'] == 'RepVGG':
        return RepVGG_8_2_align(config['backbone'])
    if config['backbone_type'] == 'RepViT':
        return RepViT_8_2(config['backbone'])
    raise ValueError(f"DGIM.BACKBONE_TYPE {config['backbone_type']} not supported.")
