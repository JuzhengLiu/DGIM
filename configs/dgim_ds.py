from src.config.default import _CN as cfg

cfg.DGIM.COARSE.TEMP_BUG_FIX = False
cfg.DGIM.MATCH_COARSE.MATCH_TYPE = 'dual_softmax'

cfg.DGIM.BACKBONE_TYPE = 'RepViT'
# Learning rates are for the configured batch, not silently rescaled by GPU count.
cfg.TRAINER.PRETRAIN_LR = 5e-4
cfg.TRAINER.FINETUNE_LR = 1.25e-4
cfg.TRAINER.WARMUP_EPOCHS = 5
cfg.TRAINER.WARMUP_RATIO = 0.1
cfg.TRAINER.PRETRAIN_MILESTONES = [13, 17, 21, 23]
cfg.TRAINER.FINETUNE_MILESTONES = [5, 8, 11, 13]

cfg.TEACHER.ENABLED = True
cfg.TEACHER.WEIGHT = 0.05
cfg.TEACHER.RAMP_EPOCHS = 5
cfg.TEACHER.STOP_EPOCH = 10

# pose estimation
cfg.TRAINER.RANSAC_PIXEL_THR = 0.5

cfg.TRAINER.OPTIMIZER = "adamw"
cfg.TRAINER.ADAMW_DECAY = 0.1
cfg.DGIM.MATCH_COARSE.TRAIN_COARSE_PERCENT = 0.3
