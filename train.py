"""Run one manually selected DGIM training stage; never starts another stage."""

import argparse

from loguru import logger
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.strategies import DDPStrategy

from src.config.training import prepare_training
from src.lightning.checkpoint import StageCheckpoint
from src.lightning.data import MultiSceneDataModule
from src.lightning.lightning_dgim import PL_DGIM


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=('pretrain', 'finetune'), required=True)
    parser.add_argument('--config', dest='stage_config', default=None,
                        help='stage YML; defaults to paths.pretrain_config or paths.finetune_config')
    parser.add_argument('--paths', dest='paths_config', default=None, help='resource path YML override')
    parser.add_argument('--model-config', '--main_cfg_path', dest='main_cfg_path', default=None)
    parser.add_argument('--data-config', '--data_cfg_path', dest='data_cfg_path', default=None)
    parser.add_argument('--backbone', choices=('repvit', 'repvgg'), default=None)
    parser.add_argument('--no-teacher', action='store_true')
    parser.add_argument('--exp-name', '--exp_name', dest='exp_name', default=None)
    parser.add_argument('--run-name', default=None, help='new experiment directory, e.g. run02')
    parser.add_argument('--batch-size', '--batch_size', dest='batch_size', type=int, default=None)
    parser.add_argument('--num-workers', '--num_workers', dest='num_workers', type=int, default=None)
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--devices', type=int, default=None)
    parser.add_argument('--num-nodes', type=int, default=None)
    parser.add_argument('--accelerator', choices=('gpu', 'cpu', 'auto'), default=None)
    parser.add_argument('--precision', choices=('32-true', '16-mixed', 'bf16-mixed'), default=None)
    parser.add_argument('--accumulate-grad-batches', type=int, default=None)
    parser.add_argument('--save-every-n-epochs', type=int, default=None)
    parser.add_argument('--pretrained', '--ckpt_path', dest='pretrained', default=None,
                        help="initialize matcher only; overrides this stage's init checkpoint path")
    parser.add_argument('--resume', '--resume_from_checkpoint', dest='resume', default=None,
                        help="resume the same stage, writing back to this checkpoint's directory")
    parser.add_argument('--disable-checkpoint', '--disable_ckpt', dest='disable_ckpt', action='store_true')
    return parser.parse_args()


def main():
    args = parse_args()
    config, paths = prepare_training(args)
    pl.seed_everything(config.TRAINER.SEED, workers=True)
    model = PL_DGIM(config, pretrained_ckpt=args.pretrained_checkpoint)
    data_module = MultiSceneDataModule(args, config)
    tensorboard = TensorBoardLogger(
        save_dir=paths['log_dir'], name=f'{args.exp_name}/{args.stage}/{args.run_name}',
        default_hp_metric=False,
    )
    callbacks = [LearningRateMonitor(logging_interval='step')]
    if config.TRAINER.EARLY_STOP_ENABLED:
        callbacks.append(EarlyStopping(
            monitor='train/epoch_loss_match', mode='min',
            patience=config.TRAINER.EARLY_STOP_PATIENCE,
            min_delta=config.TRAINER.EARLY_STOP_MIN_DELTA,
            check_on_train_epoch_end=True,
        ))
    if not args.disable_ckpt:
        callbacks.append(StageCheckpoint(
            dirpath=args.checkpoint_dir,
            every_n_epochs=config.TRAINER.SAVE_EVERY_N_EPOCHS,
            save_last=config.TRAINER.SAVE_LAST,
        ))
    distributed = args.devices * args.num_nodes > 1
    strategy = DDPStrategy(find_unused_parameters=True) if distributed else 'auto'
    trainer = pl.Trainer(
        accelerator=args.accelerator, devices=args.devices, num_nodes=args.num_nodes,
        precision=args.precision, strategy=strategy, max_epochs=args.epochs,
        accumulate_grad_batches=args.accumulate_grad_batches,
        gradient_clip_val=config.TRAINER.GRADIENT_CLIPPING,
        callbacks=callbacks, logger=tensorboard, default_root_dir=paths['log_dir'],
        enable_checkpointing=not args.disable_ckpt, use_distributed_sampler=True,
        reload_dataloaders_every_n_epochs=1,
        sync_batchnorm=distributed and args.accelerator != 'cpu',
        num_sanity_val_steps=0, log_every_n_steps=50,
    )
    logger.info(
        'Stage={} | init={} | resume={} | checkpoints={} | save every {} epochs',
        args.stage, args.pretrained_checkpoint, args.resume_checkpoint,
        args.checkpoint_dir, config.TRAINER.SAVE_EVERY_N_EPOCHS,
    )
    trainer.fit(model, datamodule=data_module, ckpt_path=args.resume_checkpoint)


if __name__ == '__main__':
    main()
