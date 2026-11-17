"""Epoch-boundary snapshots for each separately launched training stage."""

from pathlib import Path

import torch
from pytorch_lightning.callbacks import ModelCheckpoint


class StageCheckpoint(ModelCheckpoint):
    """Archive every N completed epochs and always at normal/early stopping.

    ``last.ckpt`` is an optional rolling copy updated at the same save events.
    Archived files keep both completed epoch count and optimizer step number;
    the base callback adds a suffix if that archive name already exists.
    """

    def __init__(self, dirpath, every_n_epochs, save_last):
        if not isinstance(every_n_epochs, int) or every_n_epochs < 1:
            raise ValueError("SAVE_EVERY_N_EPOCHS must be a positive integer.")
        self.save_interval = every_n_epochs
        super().__init__(
            dirpath=dirpath,
            filename="epoch={completed_epochs:03d}-step={step:07d}",
            auto_insert_metric_name=False,
            save_top_k=-1,
            save_last=save_last,
            every_n_epochs=1,
            save_on_train_epoch_end=True,
            enable_version_counter=True,
        )

    def _monitor_candidates(self, trainer):
        candidates = super()._monitor_candidates(trainer)
        candidates["completed_epochs"] = torch.tensor(trainer.current_epoch + 1)
        return candidates

    def load_state_dict(self, state_dict):
        # A moved run keeps its archive names and rolling last.ckpt on the new
        # machine. This callback has no metric-based top-k paths to relocate.
        state_dict = dict(state_dict)
        if state_dict.get("dirpath") != self.dirpath:
            state_dict["dirpath"] = self.dirpath
            for key in ("best_model_path", "last_model_path"):
                if state_dict.get(key):
                    basename = Path(state_dict[key].replace("\\", "/")).name
                    state_dict[key] = str(Path(self.dirpath) / basename)
        # The first archive is saved before last_model_path is set. Resuming
        # that archive must still update last.ckpt, rather than last-v1.ckpt.
        if self.save_last:
            state_dict["last_model_path"] = str(Path(self.dirpath) / "last.ckpt")
        super().load_state_dict(state_dict)

    def on_train_epoch_end(self, trainer, pl_module):
        completed = trainer.current_epoch + 1
        final_epoch = trainer.max_epochs > 0 and completed >= trainer.max_epochs
        if completed % self.save_interval == 0 or final_epoch or trainer.should_stop:
            super().on_train_epoch_end(trainer, pl_module)
