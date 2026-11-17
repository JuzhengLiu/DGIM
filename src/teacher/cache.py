"""Versioned, image-relative teacher caches shared by extraction and training."""

import hashlib
from pathlib import Path

import torch
import yaml

from src.datasets.preprocessing import preprocessing_spec


SCHEMA_VERSION = 2
MODEL_NAME = 'dinov3_convnext_large_pretrain_lvd1689m'
CHANNELS = 384
STRIDE = 8
ALIGNMENT = {
    'teacher_origin_xy': [3.5, 3.5],
    'student_origin_xy': [0.0, 0.0],
    'stride': STRIDE,
    'interpolation': 'bilinear',
    'align_corners': False,
    'validity': 'all_interpolation_support_inside_unpadded_stride8_cells',
}


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def image_digest(image, image_mask):
    """Bind the CPU float32 gray pixels and optional bool padding mask."""
    digest = hashlib.sha256(image.contiguous().numpy().tobytes())
    if image_mask is not None:
        digest.update(image_mask.contiguous().numpy().tobytes())
    return digest.hexdigest()


def cache_contract(img_resize, df, img_padding):
    return {
        'schema_version': SCHEMA_VERSION,
        'model': MODEL_NAME,
        'stage_index': 1,
        'channels': CHANNELS,
        'feature_stride': STRIDE,
        'feature_dtype': 'float16',
        'image_digest': 'sha256_preprocessed_cpu_float32_gray_pixels_then_optional_bool_padding_mask',
        'preprocessing': preprocessing_spec(img_resize, df, img_padding),
        'alignment': ALIGNMENT,
    }


def _write_manifest(directory, manifest):
    temporary = directory / 'manifest.yml.tmp'
    with temporary.open('w', encoding='utf-8') as stream:
        yaml.safe_dump(manifest, stream, allow_unicode=True, sort_keys=False)
    temporary.replace(directory / 'manifest.yml')


class TeacherCache:
    def __init__(self, cache_dir, data_root, img_resize=None, df=None,
                 img_padding=False, require_complete=True):
        self.directory = Path(cache_dir).resolve()
        self.data_root = Path(data_root).resolve()
        with (self.directory / 'manifest.yml').open('r', encoding='utf-8') as stream:
            self.manifest = yaml.safe_load(stream)
        expected = cache_contract(img_resize, df, img_padding)
        for key, value in expected.items():
            if self.manifest.get(key) != value:
                raise ValueError(f'Teacher cache {key} differs from this training configuration.')
        if require_complete and not self.manifest['complete']:
            raise ValueError('Teacher cache is incomplete; finish prepare_teacher_cache.py first.')

    @classmethod
    def create(cls, cache_dir, data_root, checkpoint_path, source_path, image_count,
               img_resize=None, df=None, img_padding=False):
        directory = Path(cache_dir).resolve()
        directory.mkdir(parents=True, exist_ok=True)
        manifest = cache_contract(img_resize, df, img_padding)
        manifest.update({
            'checkpoint': {
                'filename': Path(checkpoint_path).name,
                'sha256': sha256_file(checkpoint_path),
            },
            'source_sha256': sha256_file(source_path),
            'image_count': image_count,
            'complete': False,
        })
        manifest_path = directory / 'manifest.yml'
        if manifest_path.exists():
            with manifest_path.open('r', encoding='utf-8') as stream:
                previous = yaml.safe_load(stream)
            # Never silently mix another checkpoint or preprocessing in one cache.
            for key, value in manifest.items():
                if key not in {'complete', 'image_count'} and previous.get(key) != value:
                    raise ValueError(f'Existing cache has different {key}; select a new teacher_cache_dir.')
        _write_manifest(directory, manifest)
        return cls(directory, data_root, img_resize, df, img_padding, require_complete=False)

    def relative_image_path(self, image_path):
        return Path(image_path).resolve().relative_to(self.data_root).as_posix()

    def file_path(self, image_path):
        relative = self.relative_image_path(image_path)
        key = hashlib.sha256(relative.encode('utf-8')).hexdigest()
        return self.directory / 'features' / key[:2] / f'{key}.pt'

    def save(self, image_path, feature, valid_mask, image, image_mask):
        image_hw = image.shape[-2:]
        expected = (CHANNELS, image_hw[0] // STRIDE, image_hw[1] // STRIDE)
        if tuple(feature.shape) != expected or tuple(valid_mask.shape) != expected[1:]:
            raise ValueError(f'Wrong teacher feature/mask shape for {image_path}.')
        destination = self.file_path(image_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix('.pt.tmp')
        torch.save({
            'relative_image_path': self.relative_image_path(image_path),
            'image_hw': list(image_hw),
            'image_sha256': image_digest(image, image_mask),
            'feature': feature.detach().to(device='cpu', dtype=torch.float16).contiguous(),
            'valid_mask': valid_mask.detach().to(device='cpu', dtype=torch.bool).contiguous(),
        }, temporary)
        temporary.replace(destination)

    def load(self, image_path, image, image_mask):
        image_hw = image.shape[-2:]
        entry = torch.load(self.file_path(image_path), map_location='cpu', weights_only=True)
        relative = self.relative_image_path(image_path)
        expected = (CHANNELS, image_hw[0] // STRIDE, image_hw[1] // STRIDE)
        if entry['relative_image_path'] != relative or entry['image_hw'] != list(image_hw):
            raise ValueError(f'Teacher cache identity or image size differs for {relative}.')
        image_sha256 = image_digest(image, image_mask)
        if entry['image_sha256'] != image_sha256:
            raise ValueError(f'Teacher cache image content or padding mask differs for {relative}; rebuild this cache.')
        feature, mask = entry['feature'], entry['valid_mask']
        if tuple(feature.shape) != expected or tuple(mask.shape) != expected[1:]:
            raise ValueError(f'Teacher cache feature/mask shape differs for {relative}.')
        if feature.dtype != torch.float16 or mask.dtype != torch.bool:
            raise ValueError(f'Teacher cache tensor types differ for {relative}.')
        return feature, mask

    def finish(self):
        self.manifest['complete'] = True
        _write_manifest(self.directory, self.manifest)
