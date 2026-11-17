"""Build clean-gray DINOv3 ConvNeXt-Large stage-2 caches; never downloads weights."""

import argparse
from pathlib import Path

import torch

from src.config.default import get_cfg_defaults
from src.config.paths import load_paths
from src.datasets.preprocessing import enumerate_training_pairs, read_clean_gray
from src.teacher.cache import TeacherCache
from src.teacher.extract import extract_teacher_feature, load_frozen_teacher


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--paths', default=None, help='YML path overrides; same selector as train.py.')
    parser.add_argument('--device', default='cuda', help='Offline extraction device, e.g. cuda or cpu.')
    return parser.parse_args()


def main():
    args = parse_args()
    paths = load_paths(args.paths)
    config = get_cfg_defaults()
    config.merge_from_file(paths['model_config'])
    config.merge_from_file(paths['data_config'])
    root = Path(paths['train_data_root'])
    checkpoint = paths['teacher_checkpoint']
    if checkpoint is None or not Path(checkpoint).is_file():
        raise FileNotFoundError('Set paths.teacher_checkpoint to the local official ConvNeXt-Large checkpoint.')
    source = Path(paths['teacher_source_dir']) / 'convnext.py'
    if not source.is_file():
        raise FileNotFoundError(f'Teacher source not found: {source}')
    pairs = enumerate_training_pairs(root)
    images = sorted({pair[key] for pair in pairs for key in ('image0_path', 'image1_path')})
    if not images:
        raise ValueError(f'No referenced training images found in {root}.')
    resize = config.DATASET.MGDPT_IMG_RESIZE
    df = config.DATASET.MGDPT_DF
    padding = config.DATASET.MGDPT_IMG_PAD
    device = torch.device(args.device)
    model = load_frozen_teacher(paths['teacher_source_dir'], checkpoint, device)
    cache = TeacherCache.create(
        paths['teacher_cache_dir'], root, checkpoint, source, len(images),
        img_resize=resize, df=df, img_padding=padding,
    )
    print(f'Encoding {len(images)} unique training images on {device}; cache: {cache.directory}')
    for index, image_path in enumerate(images, start=1):
        image, mask, _ = read_clean_gray(image_path, resize, df, padding)
        feature, valid = extract_teacher_feature(model, image, mask, device)
        cache.save(image_path, feature, valid, image, mask)
        if index == 1 or index % 100 == 0 or index == len(images):
            print(f'{index}/{len(images)} images cached')
    cache.finish()
    print('Teacher cache complete. It contains no augmented-image targets.')


if __name__ == '__main__':
    main()
