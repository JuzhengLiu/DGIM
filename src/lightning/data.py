"""Training-only DataModule with an explicit pretrain/finetune lifecycle."""

import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader

from src.config.paths import load_paths
from src.datasets.multi_modality import MultiModality


class MultiSceneDataModule(pl.LightningDataModule):
    def __init__(self, args, config):
        super().__init__()
        self.config = config
        self.project_paths = getattr(args, "project_paths", None)
        if self.project_paths is None:
            self.project_paths = load_paths(getattr(args, "paths_config", None))
        self.stage = config.TRAINER.STAGE
        self.batch_size = args.batch_size
        self.num_workers = args.num_workers
        self.pin_memory = getattr(args, "pin_memory", True)
        self.seed = config.TRAINER.SEED
        self.teacher_enabled = (
            self.stage == "pretrain" and config.TEACHER.ENABLED and config.TEACHER.WEIGHT > 0
        )
        self.teacher_stop_epoch = config.TEACHER.STOP_EPOCH
        self.train_dataset = None
        if self.stage == "finetune" and self.num_workers != 0:
            raise ValueError("Finetuning keeps the frozen style generator in the training process; --num-workers must be 0.")

    def setup(self, stage=None):
        if stage not in (None, "fit"):
            raise ValueError("This DataModule implements training only (stage='fit').")
        if self.train_dataset is not None:
            return
        # Do not open teacher caches during setup: the restored epoch is not
        # necessarily available yet. train_dataloader enables them when needed.
        self.train_dataset = MultiModality(
            self.project_paths["train_data_root"],
            mode="train",
            img_resize=self.config.DATASET.MGDPT_IMG_RESIZE,
            df=self.config.DATASET.MGDPT_DF,
            img_padding=self.config.DATASET.MGDPT_IMG_PAD,
            style_dir=self.project_paths["style_dir"],
            style_checkpoint=self.project_paths["style_checkpoint"],
            style_device=self.trainer.strategy.root_device,
            coarse_scale=1 / self.config.DGIM.RESOLUTION[0],
            training_stage=self.stage,
            teacher_cache_dir=self.project_paths.get("teacher_cache_dir"),
            teacher_enabled=False,
        )

    def train_dataloader(self):
        if self.train_dataset is None:
            raise RuntimeError("The training dataset has not been set up.")
        epoch = max(
            self.trainer.current_epoch,
            getattr(self.trainer.lightning_module, "resume_data_epoch", 0),
        )
        self.train_dataset.set_teacher_enabled(
            self.teacher_enabled and epoch < self.teacher_stop_epoch
        )
        # Fresh workers observe the epoch's cache setting. The epoch seed also
        # prevents a resumed loader from restarting the epoch-zero shuffle.
        generator = torch.Generator().manual_seed(self.seed + epoch)
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=True,
            persistent_workers=False,
            generator=generator,
        )
