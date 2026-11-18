"""Combine model/data settings with one explicitly selected training stage."""

from pathlib import Path

from src.config.default import get_cfg_defaults
from src.config.paths import load_paths, resolve_path


def load_stage_config(paths, stage, *, model_config=None, data_config=None, stage_config=None):
    """Use the same image geometry for offline targets and their training run."""
    config = get_cfg_defaults()
    model_file = resolve_path(model_config) if model_config else paths['model_config']
    data_file = resolve_path(data_config) if data_config else paths['data_config']
    stage_file = resolve_path(stage_config) if stage_config else paths[f'{stage}_config']
    for filename in (model_file, data_file, stage_file):
        with Path(filename).open(encoding='utf-8-sig') as stream:
            config.merge_from_other_cfg(config.load_cfg(stream))
    if config.TRAINER.STAGE != stage:
        raise ValueError(f'{stage_file} must declare TRAINER.STAGE={stage}.')
    return config


def prepare_training(args):
    paths = load_paths(args.paths_config)
    config = load_stage_config(paths, args.stage, model_config=args.main_cfg_path,
                               data_config=args.data_cfg_path, stage_config=args.stage_config)

    # Command-line values override only explicitly supplied options.
    overrides = {
        'exp_name': 'EXP_NAME', 'run_name': 'RUN_NAME', 'epochs': 'MAX_EPOCHS',
        'batch_size': 'BATCH_SIZE', 'num_workers': 'NUM_WORKERS',
        'devices': 'DEVICES', 'num_nodes': 'NUM_NODES', 'accelerator': 'ACCELERATOR',
        'precision': 'PRECISION', 'accumulate_grad_batches': 'ACCUMULATE_GRAD_BATCHES',
        'save_every_n_epochs': 'SAVE_EVERY_N_EPOCHS',
    }
    for argument, key in overrides.items():
        value = getattr(args, argument)
        if value is not None:
            setattr(config.TRAINER, key, value)
    if args.backbone is not None:
        config.DGIM.BACKBONE_TYPE = {'repvit': 'RepViT', 'repvgg': 'RepVGG'}[args.backbone]
    if args.no_teacher:
        config.TEACHER.ENABLED = False
    if config.TRAINER.EXP_NAME is None:
        teacher_suffix = '_dino' if config.TEACHER.ENABLED and config.TEACHER.WEIGHT > 0 else ''
        config.TRAINER.EXP_NAME = config.DGIM.BACKBONE_TYPE.lower() + teacher_suffix
    for argument, key in overrides.items():
        setattr(args, argument, getattr(config.TRAINER, key))
    for name in (args.exp_name, args.run_name):
        if not name or name in ('.', '..') or any(character in name for character in ('/', '\\')):
            raise ValueError('EXP_NAME and RUN_NAME must each be a single directory name.')
    if min(args.epochs, args.batch_size, args.devices, args.num_nodes,
           args.accumulate_grad_batches, args.save_every_n_epochs) < 1:
        raise ValueError('Epochs, batch size, device counts, accumulation and save interval must be positive.')
    if args.stage == 'finetune' and args.num_workers != 0:
        raise ValueError('Finetuning uses the style generator in the training process; set NUM_WORKERS: 0.')
    config.TRAINER.WORLD_SIZE = args.devices * args.num_nodes
    config.DATASET.TRAIN_DATA_ROOT = paths['train_data_root']
    args.project_paths = paths

    args.resume_checkpoint = resolve_path(args.resume) if args.resume else paths[f'{args.stage}_resume_checkpoint']
    if args.resume_checkpoint:
        if args.pretrained:
            raise ValueError('Use --resume for continuation or --pretrained for initialization, not both.')
        args.pretrained_checkpoint = None
        # Continue exactly this run, including after moving its directory to a server.
        checkpoint_dir = Path(args.resume_checkpoint).parent
        args.run_name = config.TRAINER.RUN_NAME = checkpoint_dir.name
    else:
        args.pretrained_checkpoint = (
            resolve_path(args.pretrained) if args.pretrained else paths[f'{args.stage}_init_checkpoint']
        )
        if args.stage == 'finetune' and args.pretrained_checkpoint is None:
            raise ValueError('Set paths.finetune_init_checkpoint or pass --pretrained for stage two.')
        checkpoint_dir = Path(paths['checkpoint_dir']) / args.exp_name / args.stage / args.run_name
        # A fresh run must never overwrite earlier training. Resume is explicit.
        if not args.disable_ckpt and any(checkpoint_dir.glob('*.ckpt')):
            raise FileExistsError(
                f'{checkpoint_dir} already contains checkpoints. Select a new RUN_NAME or use --resume.'
            )
    args.checkpoint_dir = checkpoint_dir
    for filename in (args.pretrained_checkpoint, args.resume_checkpoint):
        if filename is not None and not Path(filename).is_file():
            raise FileNotFoundError(filename)
    return config, paths
