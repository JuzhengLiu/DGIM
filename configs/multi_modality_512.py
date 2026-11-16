from configs.base import cfg

# Filesystem locations are supplied by configs/paths.json in train.py.
# The released DataModule only implements training; no MegaDepth test set is bundled.
cfg.DATASET.TRAINVAL_DATA_SOURCE = "Multi-Modality"
cfg.DATASET.TEST_DATA_SOURCE = None

cfg.DATASET.MGDPT_IMG_RESIZE = None
cfg.DATASET.MGDPT_IMG_PAD = True
cfg.DATASET.MGDPT_DF = 8
cfg.DATASET.MIN_OVERLAP_SCORE_TEST = 0.0
