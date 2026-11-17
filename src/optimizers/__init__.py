"""Optimizers and a resumable learning-rate schedule for Lightning 2.x."""

import math

import torch
from torch.optim.lr_scheduler import LRScheduler


def build_optimizer(model, config):
    name = config.TRAINER.OPTIMIZER.lower()
    lr = config.TRAINER.LR
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if name == "adam":
        return torch.optim.Adam(parameters, lr=lr, weight_decay=config.TRAINER.ADAM_DECAY)
    if name == "adamw":
        return torch.optim.AdamW(parameters, lr=lr, weight_decay=config.TRAINER.ADAMW_DECAY)
    raise ValueError(f"Unsupported TRAINER.OPTIMIZER: {name}")


class WarmupMultiStepLR(LRScheduler):
    """Count optimizer updates, using the real training loader's epoch length.

    Lightning constructs optimizers before the training loader is ready. The
    initial learning rate does not depend on its length; configure that length
    in on_train_start, after Lightning has restored its training loop. Both
    the length and update counter are saved in the scheduler state dict.
    """

    def __init__(self, optimizer, warmup_epochs, warmup_ratio, milestones, gamma):
        if warmup_epochs < 0 or not 0 < warmup_ratio <= 1:
            raise ValueError("Warmup requires nonnegative epochs and a ratio in (0, 1].")
        if not 0 < gamma <= 1:
            raise ValueError("MSLR_GAMMA must be in (0, 1].")
        if any(int(epoch) != epoch or epoch < 0 for epoch in milestones):
            raise ValueError("Learning-rate milestones must be nonnegative integer epochs.")
        self.warmup_epochs = float(warmup_epochs)
        self.warmup_ratio = float(warmup_ratio)
        self.milestones = tuple(sorted(int(epoch) for epoch in milestones))
        self.gamma = float(gamma)
        self.steps_per_epoch = 1
        self.steps_per_epoch_ready = False
        super().__init__(optimizer)

    def configure_steps_per_epoch(self, steps_per_epoch):
        if not isinstance(steps_per_epoch, int) or steps_per_epoch < 1:
            raise ValueError("The training loader must provide at least one optimizer step per epoch.")
        if self.steps_per_epoch_ready and self.steps_per_epoch != steps_per_epoch:
            raise ValueError(
                "Resume would change optimizer steps per epoch "
                f"({self.steps_per_epoch} -> {steps_per_epoch}). Keep the dataset, "
                "batch size, device count and accumulation unchanged, or start a "
                "new run with --pretrained instead of --resume."
            )
        self.steps_per_epoch = steps_per_epoch
        self.steps_per_epoch_ready = True

    def get_lr(self):
        progress = max(self.last_epoch, 0) / self.steps_per_epoch
        if self.warmup_epochs and progress < self.warmup_epochs:
            factor = self.warmup_ratio + (1.0 - self.warmup_ratio) * progress / self.warmup_epochs
        else:
            factor = 1.0
        factor *= self.gamma ** sum(progress >= epoch for epoch in self.milestones)
        return [base_lr * factor for base_lr in self.base_lrs]


def build_scheduler(config, optimizer):
    scheduler = WarmupMultiStepLR(
        optimizer,
        warmup_epochs=config.TRAINER.WARMUP_EPOCHS,
        warmup_ratio=config.TRAINER.WARMUP_RATIO,
        milestones=config.TRAINER.MILESTONES,
        gamma=config.TRAINER.MSLR_GAMMA,
    )
    return {"scheduler": scheduler, "interval": "step", "frequency": 1,
            "name": "learning_rate"}


def optimizer_steps_per_epoch(num_training_batches, accumulate_grad_batches):
    if not math.isfinite(num_training_batches) or num_training_batches <= 0:
        raise ValueError("A finite, nonempty training loader is required.")
    return math.ceil(num_training_batches / accumulate_grad_batches)
