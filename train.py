"""Train DGIM in explicit pretrain/finetune stages with Lightning 2.2."""

import argparse
from pathlib import Path

from loguru import logger
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.strategies import DDPStrategy

from src.config.default import get_cfg_defaults
from src.config.paths import load_paths, resolve_path
from src.lightning.data import MultiSceneDataModule
from src.lightning.lightning_dgim import PL_DGIM
from src.losses.teacher_loss import teacher_weight


def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--stage", choices=("pretrain", "finetune"), default=None,
                        help="training stage; otherwise use TRAINER.STAGE")
    parser.add_argument("--backbone", choices=("repvit", "repvgg"), default=None,
                        help="backbone; otherwise use the model configuration")
    parser.add_argument("--no-teacher", action="store_true", help="disable cached teacher supervision")
    parser.add_argument("--paths", dest="paths_config", default=None,
                        help="path configuration overlay")
    parser.add_argument("--model-config", "--main_cfg_path", dest="main_cfg_path", default=None)
    parser.add_argument("--data-config", "--data_cfg_path", dest="data_cfg_path", default=None)
    parser.add_argument("--exp-name", "--exp_name", dest="exp_name", default=None)
    parser.add_argument("--batch-size", "--batch_size", dest="batch_size", type=int, default=4,
                        help="batch size per device; stage learning rates are not auto-scaled")
    parser.add_argument("--num-workers", "--num_workers", dest="num_workers", type=int, default=0,
                        help="finetuning requires 0 because style generation uses CUDA")
    parser.add_argument("--epochs", type=int, default=None, help="override this stage's maximum epochs")
    parser.add_argument("--devices", type=int, default=1, help="number of devices per node")
    parser.add_argument("--num-nodes", type=int, default=1)
    parser.add_argument("--accelerator", choices=("gpu", "cpu", "auto"), default="gpu")
    parser.add_argument("--precision", choices=("32-true", "16-mixed", "bf16-mixed"), default="32-true")
    parser.add_argument("--accumulate-grad-batches", type=int, default=1)
    parser.add_argument("--pretrained", "--ckpt_path", dest="pretrained", default=None,
                        help="initialize matcher only; finetune defaults to paths.pretrained_checkpoint")
    parser.add_argument("--resume", "--resume_from_checkpoint", dest="resume", default=None,
                        help="resume model, optimizer, schedule and training loop")
    parser.add_argument("--disable-checkpoint", "--disable_ckpt", dest="disable_ckpt", action="store_true")
    return parser.parse_args()


def _existing_file(label, filename):
    if filename is None or not Path(filename).is_file():
        raise FileNotFoundError(
            f"{label} not found: {filename}. Set the corresponding path configuration or CLI override."
        )


def prepare_configuration(args):
    paths = load_paths(args.paths_config)
    args.project_paths = paths
    args.main_cfg_path = resolve_path(args.main_cfg_path) if args.main_cfg_path else paths["model_config"]
    args.data_cfg_path = resolve_path(args.data_cfg_path) if args.data_cfg_path else paths["data_config"]
    _existing_file("model_config", args.main_cfg_path)
    _existing_file("data_config", args.data_cfg_path)
    config = get_cfg_defaults()
    config.merge_from_file(args.main_cfg_path)
    config.merge_from_file(args.data_cfg_path)
    for split in ("train", "val", "test"):
        for suffix in ("data_root", "pose_root", "npz_root", "list_path", "intrinsic_path"):
            key = f"{split}_{suffix}"
            setattr(config.DATASET, key.upper(), paths[key])
    if args.stage is not None:
        config.TRAINER.STAGE = args.stage
    args.stage = config.TRAINER.STAGE
    if args.stage not in ("pretrain", "finetune"):
        raise ValueError("TRAINER.STAGE must be pretrain or finetune.")
    if args.backbone is not None:
        config.DGIM.BACKBONE_TYPE = {"repvit": "RepViT", "repvgg": "RepVGG"}[args.backbone]
    if config.DGIM.BACKBONE_TYPE not in ("RepViT", "RepVGG"):
        raise ValueError("DGIM.BACKBONE_TYPE must be RepViT or RepVGG.")
    args.backbone = config.DGIM.BACKBONE_TYPE.lower()
    if args.no_teacher:
        config.TEACHER.ENABLED = False
    if args.exp_name is None:
        # Keep the same experiment directory in finetuning even though the
        # teacher is only active during pretraining.
        use_teacher_name = config.TEACHER.ENABLED and config.TEACHER.WEIGHT > 0
        args.exp_name = args.backbone + ("_dino" if use_teacher_name else "")
    if not args.exp_name or args.exp_name in (".", "..") or any(
        character in args.exp_name for character in ("/", "\\")
    ):
        raise ValueError("--exp-name must be a single nonempty directory name.")
    if min(args.batch_size, args.devices, args.num_nodes, args.accumulate_grad_batches) < 1:
        raise ValueError("Batch size, devices, nodes and gradient accumulation must be positive.")
    if args.num_workers < 0:
        raise ValueError("--num-workers cannot be negative.")
    if args.stage == "finetune" and args.num_workers != 0:
        raise ValueError("Finetuning requires --num-workers 0 for the frozen style generator.")
    epoch_key = "PRETRAIN_EPOCHS" if args.stage == "pretrain" else "FINETUNE_EPOCHS"
    if args.epochs is not None:
        setattr(config.TRAINER, epoch_key, args.epochs)
    args.epochs = getattr(config.TRAINER, epoch_key)
    if args.epochs < 1:
        raise ValueError("--epochs must be positive.")
    config.TRAINER.WORLD_SIZE = args.devices * args.num_nodes
    config.TRAINER.TRUE_LR = (
        config.TRAINER.PRETRAIN_LR if args.stage == "pretrain" else config.TRAINER.FINETUNE_LR
    )
    if config.TRAINER.TRUE_LR <= 0:
        raise ValueError("The stage learning rate must be positive.")
    teacher_weight(
        0, enabled=(args.stage == "pretrain" and config.TEACHER.ENABLED),
        weight=config.TEACHER.WEIGHT, ramp_epochs=config.TEACHER.RAMP_EPOCHS,
        stop_epoch=config.TEACHER.STOP_EPOCH,
    )
    args.resume_checkpoint = resolve_path(args.resume) if args.resume else paths["resume_checkpoint"]
    args.pretrained_checkpoint = resolve_path(args.pretrained) if args.pretrained else None
    if args.resume_checkpoint and args.pretrained_checkpoint:
        raise ValueError("Choose either --resume or --pretrained, not both.")
    if args.stage == "finetune" and not args.resume_checkpoint and not args.pretrained_checkpoint:
        args.pretrained_checkpoint = paths["pretrained_checkpoint"]
        if args.pretrained_checkpoint is None:
            raise ValueError(
                "Finetuning requires a pretraining checkpoint. Set paths.pretrained_checkpoint, "
                "pass --pretrained, or resume an existing finetuning run with --resume."
            )
    for label, filename in (("resume_checkpoint", args.resume_checkpoint),
                            ("pretrained_checkpoint", args.pretrained_checkpoint)):
        if filename is not None:
            _existing_file(label, filename)
    # Fresh pretraining deliberately ignores paths.pretrained_checkpoint so a
    # baseline starts from random initialization unless --pretrained is explicit.
    return config, paths


def main():
    args = parse_args()
    config, paths = prepare_configuration(args)
    pl.seed_everything(config.TRAINER.SEED, workers=True)
    model = PL_DGIM(config, pretrained_ckpt=args.pretrained_checkpoint)
    data_module = MultiSceneDataModule(args, config)
    tensorboard = TensorBoardLogger(
        save_dir=paths["log_dir"], name=f"{args.exp_name}/{args.stage}", default_hp_metric=False
    )
    checkpoint_root = Path(paths["checkpoint_dir"] or paths["log_dir"])
    checkpoint_dir = checkpoint_root / args.exp_name / args.stage
    callbacks = [LearningRateMonitor(logging_interval="step")]
    if not args.disable_ckpt:
        callbacks.append(ModelCheckpoint(
            dirpath=checkpoint_dir,
            filename="epoch={epoch:02d}",
            auto_insert_metric_name=False,
            save_top_k=-1,
            save_last=True,
            every_n_epochs=1,
            save_on_train_epoch_end=True,
            enable_version_counter=False,
        ))
    if args.stage == "finetune":
        callbacks.append(EarlyStopping(
            monitor="train/epoch_loss",
            mode="min",
            patience=config.TRAINER.EARLY_STOP_PATIENCE,
            min_delta=config.TRAINER.EARLY_STOP_MIN_DELTA,
            check_on_train_epoch_end=True,
            check_finite=True,
            strict=True,
        ))
    distributed = args.devices * args.num_nodes > 1
    # The projection becomes unused after teacher stop_epoch, and matching can
    # have empty supervision branches; DDP must support unused parameters.
    strategy = DDPStrategy(find_unused_parameters=True) if distributed else "auto"
    trainer = pl.Trainer(
        accelerator=args.accelerator,
        devices=args.devices,
        num_nodes=args.num_nodes,
        precision=args.precision,
        strategy=strategy,
        max_epochs=args.epochs,
        accumulate_grad_batches=args.accumulate_grad_batches,
        gradient_clip_val=config.TRAINER.GRADIENT_CLIPPING,
        callbacks=callbacks,
        logger=tensorboard,
        default_root_dir=paths["log_dir"],
        enable_checkpointing=not args.disable_ckpt,
        use_distributed_sampler=True,
        reload_dataloaders_every_n_epochs=1,
        sync_batchnorm=distributed and args.accelerator != "cpu",
        num_sanity_val_steps=0,
        log_every_n_steps=50,
    )
    logger.info(
        "Starting {}: backbone={}, teacher during pretraining={}, checkpoints={}",
        args.stage, config.DGIM.BACKBONE_TYPE, config.TEACHER.ENABLED, checkpoint_dir,
    )
    trainer.fit(model, datamodule=data_module, ckpt_path=args.resume_checkpoint)


if __name__ == "__main__":
    main()
