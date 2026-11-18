from yacs.config import CfgNode as CN

# YACS merges only these model overrides into a fresh default configuration.
cfg = CN()
cfg.DGIM = CN()
cfg.DGIM.COARSE = CN()
cfg.DGIM.MATCH_COARSE = CN()

cfg.DGIM.COARSE.TEMP_BUG_FIX = False
cfg.DGIM.MATCH_COARSE.MATCH_TYPE = 'dual_softmax'

cfg.DGIM.BACKBONE_TYPE = 'RepViT'
# Training and teacher schedules are configured independently per stage in YML.
cfg.DGIM.MATCH_COARSE.TRAIN_COARSE_PERCENT = 0.3
