"""DGIM training on Lightning 2.x with optional cached feature supervision."""

from collections.abc import Mapping

from loguru import logger
import pytorch_lightning as pl
from pytorch_lightning.profilers import PassThroughProfiler
import torch
from torch import nn

from src.dgim import DGIM
from src.dgim.utils.supervision import compute_supervision_coarse, compute_supervision_fine
from src.losses.dgim_loss import DGIMLoss
from src.losses.teacher_loss import masked_cosine_feature_loss, teacher_weight
from src.optimizers import build_optimizer, build_scheduler, optimizer_steps_per_epoch


def config_to_dict(value, lowercase=False):
    """Save configuration metadata as plain dictionaries, without YACS objects."""
    if isinstance(value, Mapping):
        return {
            str(key).lower() if lowercase else key: config_to_dict(item, lowercase)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [config_to_dict(item, lowercase) for item in value]
    return value


class PL_DGIM(pl.LightningModule):
    def __init__(self, config, pretrained_ckpt=None, profiler=None, dump_dir=None):
        super().__init__()
        self.config = config
        self.training_config = config_to_dict(config)
        lower_config = config_to_dict(config, lowercase=True)
        self.dgim_cfg = lower_config["dgim"]
        self.profiler = profiler or PassThroughProfiler()
        self.matcher = DGIM(config=self.dgim_cfg)
        self.loss = DGIMLoss(lower_config)
        self.dump_dir = dump_dir
        self.initialization_checkpoint = pretrained_ckpt
        self.teacher_enabled = (
            config.TRAINER.STAGE == "pretrain"
            and config.TEACHER.ENABLED
            and config.TEACHER.WEIGHT > 0
        )
        # Only pretraining with a teacher has projection parameters. It maps
        # the student's raw /8 features to the fixed cached teacher dimension.
        self.teacher_projection = None
        if self.teacher_enabled:
            self.teacher_projection = nn.Conv2d(
                config.DGIM.COARSE.D_MODEL, config.TEACHER.DIM, kernel_size=1, bias=False
            )
            nn.init.kaiming_normal_(self.teacher_projection.weight, mode="fan_out", nonlinearity="linear")
        self._lr_scheduler = None
        self._resume_runtime = None
        self.resume_data_epoch = 0
        self._epoch_totals = None
        if pretrained_ckpt:
            self._load_pretrained_matcher(pretrained_ckpt)

    def _load_pretrained_matcher(self, filename):
        checkpoint = torch.load(filename, map_location="cpu", weights_only=False)
        if not isinstance(checkpoint, Mapping) or not isinstance(checkpoint.get("state_dict"), Mapping):
            raise ValueError("--pretrained requires a Lightning checkpoint containing state_dict.")
        state = {
            name[len("matcher."):]: tensor
            for name, tensor in checkpoint["state_dict"].items()
            if name.startswith("matcher.")
        }
        if not state:
            raise ValueError("The pretrained checkpoint contains no matcher.* parameters.")
        saved_model = checkpoint.get("dgim_config")
        if saved_model is not None:
            source_backbone = saved_model.get("backbone_type")
            if source_backbone != self.dgim_cfg["backbone_type"]:
                raise ValueError(
                    f"Cannot initialize {self.dgim_cfg['backbone_type']} with a "
                    f"{source_backbone} matcher checkpoint."
                )
        try:
            self.matcher.load_state_dict(state, strict=True)
        except RuntimeError as error:
            raise ValueError(
                "The pretrained matcher architecture does not match this run. "
                "Use a checkpoint from the same backbone and block configuration; "
                "RepVGG weights cannot initialize the custom RepViT."
            ) from error
        logger.info("Loaded matcher parameters from {}; optimizer and teacher projection start fresh.", filename)

    def configure_optimizers(self):
        optimizer = build_optimizer(self, self.config)
        scheduler_config = build_scheduler(self.config, optimizer)
        self._lr_scheduler = scheduler_config["scheduler"]
        return {"optimizer": optimizer, "lr_scheduler": scheduler_config}

    def _runtime_metadata(self):
        return {
            "batch_size": self.trainer.datamodule.batch_size,
            "world_size": self.trainer.world_size,
            "accumulate_grad_batches": self.trainer.accumulate_grad_batches,
            "num_training_batches": int(self.trainer.num_training_batches),
            "steps_per_epoch": self._lr_scheduler.steps_per_epoch,
        }

    def on_train_start(self):
        steps = optimizer_steps_per_epoch(
            self.trainer.num_training_batches, self.trainer.accumulate_grad_batches
        )
        self._lr_scheduler.configure_steps_per_epoch(steps)
        if self._resume_runtime is not None:
            current = self._runtime_metadata()
            for key in ("batch_size", "world_size", "accumulate_grad_batches",
                        "num_training_batches", "steps_per_epoch"):
                if self._resume_runtime.get(key) != current[key]:
                    raise ValueError(
                        f"--resume requires unchanged {key}: "
                        f"{self._resume_runtime.get(key)} != {current[key]}. "
                        "Use --pretrained to start a new schedule."
                    )
        logger.info(
            "Stage={} | optimizer steps/epoch={} | teacher={} | LR={}",
            self.config.TRAINER.STAGE, steps, self.teacher_enabled,
            self._lr_scheduler.get_last_lr()[0],
        )

    def on_save_checkpoint(self, checkpoint):
        checkpoint["dgim_config"] = config_to_dict(self.dgim_cfg)
        checkpoint["training_config"] = config_to_dict(self.training_config)
        checkpoint["training_runtime"] = self._runtime_metadata()
        checkpoint_callback = self.trainer.checkpoint_callback
        checkpoint["training_run"] = {
            "stage": self.config.TRAINER.STAGE,
            "run_name": self.config.TRAINER.RUN_NAME,
            "initialization_checkpoint": self.initialization_checkpoint,
            "checkpoint_dir": (
                str(checkpoint_callback.dirpath) if checkpoint_callback is not None else None
            ),
            "log_dir": self.trainer.log_dir,
        }

    def on_load_checkpoint(self, checkpoint):
        self.initialization_checkpoint = checkpoint.get("training_run", {}).get("initialization_checkpoint")
        saved = checkpoint.get("training_config")
        if not isinstance(saved, Mapping):
            raise ValueError(
                "This checkpoint has no training metadata for a safe Lightning 2 resume. "
                "Use --pretrained to load its matcher and start a new run."
            )
        saved_trainer = saved.get("TRAINER", {})
        if "LR" not in saved_trainer:
            raise ValueError(
                "This checkpoint uses the v4 training schedule. Load its matcher with "
                "--pretrained in a new run; v5 --resume requires v5 training metadata."
            )
        for section in ("DGIM", "TEACHER"):
            if saved.get(section) != self.training_config[section]:
                raise ValueError(f"--resume requires the same {section} configuration as the checkpoint.")
        # Maximum epochs may be extended. Stage, optimization, and distillation
        # settings must remain unchanged to preserve the resumed schedule.
        trainer_keys = (
            "STAGE", "OPTIMIZER", "LR", "ADAM_DECAY",
            "ADAMW_DECAY", "WARMUP_EPOCHS", "WARMUP_RATIO",
            "MILESTONES", "MSLR_GAMMA",
            "GRADIENT_CLIPPING", "SEED",
        )
        for key in trainer_keys:
            if saved_trainer.get(key) != self.training_config["TRAINER"][key]:
                raise ValueError(f"--resume requires unchanged TRAINER.{key}.")
        for key in ("MGDPT_IMG_RESIZE", "MGDPT_IMG_PAD", "MGDPT_DF"):
            if saved.get("DATASET", {}).get(key) != self.training_config["DATASET"][key]:
                raise ValueError(f"--resume requires unchanged DATASET.{key}.")
        self._resume_runtime = checkpoint.get("training_runtime")
        if not isinstance(self._resume_runtime, Mapping):
            raise ValueError("--resume requires training_runtime metadata from this training entry point.")
        # Lightning 2.2 creates the first loader before on_run_start updates
        # current_epoch after an epoch-end restore. At that save boundary,
        # processed already counts the finished epoch but completed does not.
        progress = (checkpoint.get("loops", {}).get("fit_loop", {})
                    .get("epoch_progress", {}).get("current", {}))
        processed = progress.get("processed")
        if (isinstance(processed, int) and processed >= 0
                and progress.get("ready") == processed
                and progress.get("started") == processed):
            self.resume_data_epoch = processed

    def _teacher_weight(self):
        return teacher_weight(
            self.current_epoch,
            enabled=self.teacher_enabled,
            weight=self.config.TEACHER.WEIGHT,
            ramp_epochs=self.config.TEACHER.RAMP_EPOCHS,
            stop_epoch=self.config.TEACHER.STOP_EPOCH,
        )

    def _trainval_inference(self, batch, return_backbone=False):
        device_type = batch["image0"].device.type
        with self.profiler.profile("Compute coarse supervision"):
            with torch.autocast(device_type=device_type, enabled=False):
                compute_supervision_coarse(batch, self.config)
        with self.profiler.profile("DGIM"):
            self.matcher(batch, return_backbone=return_backbone)
        with self.profiler.profile("Compute fine supervision"):
            with torch.autocast(device_type=device_type, enabled=False):
                compute_supervision_fine(batch, self.config)
        with self.profiler.profile("Compute matching losses"):
            self.loss(batch)

    def on_train_epoch_start(self):
        # Keep only detached sums, rather than an epoch of output/graph tensors.
        self._epoch_totals = torch.zeros(7, device=self.device, dtype=torch.float64)

    def training_step(self, batch, batch_idx):
        weight = self._teacher_weight()
        self._trainval_inference(batch, return_backbone=weight > 0)
        matching_loss = batch["loss"]
        feature_loss = matching_loss.new_zeros(())
        if weight > 0:
            losses = []
            for view in (0, 1):
                feature_key = f"teacher_feature{view}"
                mask_key = f"teacher_mask{view}"
                if feature_key not in batch or mask_key not in batch:
                    raise ValueError(
                        "Teacher supervision is active but cached features are absent. "
                        "Prepare the teacher cache, or start a separate --no-teacher run."
                    )
                prediction = self.teacher_projection(batch[f"backbone_c{view}"])
                losses.append(masked_cosine_feature_loss(
                    prediction, batch[feature_key], batch[mask_key]
                ))
            feature_loss = 0.5 * (losses[0] + losses[1])
        total_loss = matching_loss + weight * feature_loss
        batch_size = batch["image0"].shape[0]
        metrics = {
            "loss": total_loss,
            "loss_match": matching_loss,
            "loss_c": batch["loss_c"],
            "loss_f": batch["loss_f"],
            "loss_sub": batch["loss_sub"],
            "loss_teacher": feature_loss,
        }
        self.log_dict(
            {f"train/{name}": value.detach() for name, value in metrics.items()},
            on_step=True, on_epoch=False, batch_size=batch_size, sync_dist=False,
        )
        self.log("train/teacher_weight", weight, on_step=True, on_epoch=False,
                 batch_size=batch_size)
        values = torch.stack([value.detach().double() for value in metrics.values()])
        self._epoch_totals[:6] += values * batch_size
        self._epoch_totals[6] += batch_size
        return total_loss

    def on_train_epoch_end(self):
        # Reduce sums and sample counts, so DDP logs the true weighted mean.
        totals = self.trainer.strategy.reduce(self._epoch_totals, reduce_op="sum")
        means = totals[:6] / totals[6].clamp_min(1)
        names = ("loss", "loss_match", "loss_c", "loss_f", "loss_sub", "loss_teacher")
        self.log_dict(
            {f"train/epoch_{name}": value for name, value in zip(names, means)},
            on_step=False, on_epoch=True, sync_dist=False, batch_size=1,
        )
        self._epoch_totals = None
