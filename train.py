import torch.multiprocessing as mp
import argparse
import pprint
from distutils.util import strtobool
from pathlib import Path
from loguru import logger as loguru_logger
import os

import pytorch_lightning as pl
from pytorch_lightning.utilities import rank_zero_only
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.plugins import DDPPlugin

from src.config.default import get_cfg_defaults
from src.config.paths import load_paths, resolve_path
from src.utils.misc import get_rank_zero_only_logger, setup_gpus
from src.utils.profiler import build_profiler
from src.lightning.data import MultiSceneDataModule
from src.lightning.lightning_dgim import PL_DGIM

loguru_logger = get_rank_zero_only_logger(loguru_logger)

os.environ["PL_TORCH_DISTRIBUTED_BACKEND"] = "gloo"
os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"


def parse_args():
    # init a costum parser which will be added into pl.Trainer parser
    # check documentation: https://pytorch-lightning.readthedocs.io/en/latest/common/trainer.html#trainer-flags
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '--paths', dest='paths_config', type=str, default=None,
        help='YML path overrides; default: configs/paths.yml plus optional paths.local.yml')
    parser.add_argument(
        '--data_cfg_path', type=str, default=None,
        help='override paths.data_config (relative to the repository root)')
    parser.add_argument(
        '--main_cfg_path', type=str, default=None,
        help='override paths.model_config (relative to the repository root)')
    parser.add_argument(
        '--exp_name', type=str, default='DGIM')
    parser.add_argument(
        '--batch_size', type=int, default=4, help='batch_size per gpu')
    parser.add_argument(
        '--num_workers', type=int, default=4)
    parser.add_argument(
        '--pin_memory', type=lambda x: bool(strtobool(x)),
        nargs='?', default=True, help='whether loading data to pinned memory or not')
    parser.add_argument(
        '--ckpt_path', type=str, default=None,
        help='pretrained checkpoint path, helpful for using a pre-trained coarse-only DGIM')
    parser.add_argument(
        '--disable_ckpt', action='store_true',
        help='disable checkpoint saving (useful for debugging).')
    parser.add_argument(
        '--profiler_name', type=str, default=None,
        help='options: [inference, pytorch], or leave it unset')
    parser.add_argument(
        '--parallel_load_data', action='store_true',
        help='load datasets in with multiple processes.')

    parser = pl.Trainer.add_argparse_args(parser)
    return parser.parse_args()


def main():
    # parse arguments
    args = parse_args()
    paths = load_paths(args.paths_config)
    args.data_cfg_path = resolve_path(args.data_cfg_path) if args.data_cfg_path else paths['data_config']
    args.main_cfg_path = resolve_path(args.main_cfg_path) if args.main_cfg_path else paths['model_config']
    args.ckpt_path = resolve_path(args.ckpt_path) if args.ckpt_path else paths['pretrained_checkpoint']
    args.project_paths = paths
    args.default_root_dir = paths['log_dir']
    if getattr(args, 'resume_from_checkpoint', None):
        args.resume_from_checkpoint = resolve_path(args.resume_from_checkpoint)
    else:
        args.resume_from_checkpoint = paths['resume_checkpoint']
    for label, filename in [('data_config', args.data_cfg_path), ('model_config', args.main_cfg_path),
                            ('pretrained_checkpoint', args.ckpt_path),
                            ('resume_checkpoint', args.resume_from_checkpoint)]:
        if filename is not None and not Path(filename).is_file():
            raise FileNotFoundError(f'{label} not found: {filename}. Update the path configuration or CLI override.')
    args.gpus = -1
    rank_zero_only(pprint.pprint)(vars(args))

    # init default-cfg and merge it with the main- and data-cfg
    config = get_cfg_defaults()
    config.merge_from_file(args.main_cfg_path)
    config.merge_from_file(args.data_cfg_path)
    # Filesystem locations have one source of truth, independent of model settings.
    for split in ('train', 'val', 'test'):
        for suffix in ('data_root', 'pose_root', 'npz_root', 'list_path', 'intrinsic_path'):
            key = f'{split}_{suffix}'
            setattr(config.DATASET, key.upper(), paths[key])
    pl.seed_everything(config.TRAINER.SEED)  # reproducibility
    # TODO: Use different seeds for each dataloader workers
    # This is needed for data augmentation

    # scale lr and warmup-step automatically
    args.gpus = _n_gpus = setup_gpus(args.gpus)
    config.TRAINER.WORLD_SIZE = _n_gpus * args.num_nodes
    config.TRAINER.TRUE_BATCH_SIZE = config.TRAINER.WORLD_SIZE * args.batch_size
    _scaling = config.TRAINER.TRUE_BATCH_SIZE / config.TRAINER.CANONICAL_BS
    config.TRAINER.SCALING = _scaling
    config.TRAINER.TRUE_LR = config.TRAINER.CANONICAL_LR * _scaling
    # config.TRAINER.WARMUP_STEP = math.floor(config.TRAINER.WARMUP_STEP / _scaling)

    # ----- pretrain -----
    config.TRAINER.WARMUP_STEP = 1798 * 5  # step_number_per_epoch * epoch_number
    config.TRAINER.MSLR_MILESTONES = [13, 17, 21, 23]  # default: [10, 15, 20, 25]

    # ----- finetune -----
    # config.TRAINER.TRUE_LR /= 4
    # config.TRAINER.WARMUP_STEP = 0
    # config.TRAINER.MSLR_MILESTONES = [5, 8, 11, 13]  # default: [3, 6, 9, 12]

    # lightning module
    profiler = build_profiler(args.profiler_name)
    model = PL_DGIM(config, pretrained_ckpt=args.ckpt_path, profiler=profiler)
    loguru_logger.info(f"DGIM LightningModule initialized!")

    # lightning data
    data_module = MultiSceneDataModule(args, config)
    loguru_logger.info(f"DGIM DataModule initialized!")

    # TensorBoard Logger
    logger = TensorBoardLogger(save_dir=paths['log_dir'], name=args.exp_name, default_hp_metric=False)
    ckpt_dir = (Path(paths['checkpoint_dir']) / args.exp_name / f'version_{logger.version}'
                if paths['checkpoint_dir'] else Path(logger.log_dir) / 'checkpoints')

    # Callbacks
    # TODO: update ModelCheckpoint to monitor multiple metrics
    ckpt_callback = ModelCheckpoint(dirpath=str(ckpt_dir), filename='{epoch:02d}', save_top_k=-1)
    lr_monitor = LearningRateMonitor(logging_interval='step')
    callbacks = [lr_monitor]
    if not args.disable_ckpt:
        callbacks.append(ckpt_callback)

    # Lightning Trainer
    trainer = pl.Trainer.from_argparse_args(
        args,
        plugins=DDPPlugin(find_unused_parameters=True,
                          num_nodes=args.num_nodes,
                          sync_batchnorm=config.TRAINER.WORLD_SIZE > 0),
        gradient_clip_val=config.TRAINER.GRADIENT_CLIPPING,
        callbacks=callbacks,
        logger=logger,
        sync_batchnorm=config.TRAINER.WORLD_SIZE > 0,
        replace_sampler_ddp=False,  # use custom sampler
        reload_dataloaders_every_epoch=False,  # avoid repeated samples!
        weights_summary='full',
        profiler=profiler,
        max_epochs=25)  # set the max number of epoch
    loguru_logger.info(f"Trainer initialized!")
    loguru_logger.info(f"Start training!")
    trainer.fit(model, datamodule=data_module)


if __name__ == '__main__':
    mp.set_start_method('spawn')
    main()
