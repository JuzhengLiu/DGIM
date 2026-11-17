from src.config.default import _CN as cfg

cfg.DGIM.COARSE.TEMP_BUG_FIX = False
cfg.DGIM.MATCH_COARSE.MATCH_TYPE = 'dual_softmax'

cfg.DGIM.BACKBONE_TYPE = 'RepViT'
# Training and teacher schedules are configured independently per stage in YML.
cfg.DGIM.MATCH_COARSE.TRAIN_COARSE_PERCENT = 0.3
