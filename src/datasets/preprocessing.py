"""Shared image geometry for DGIM inputs and offline teacher caches."""

from pathlib import Path

import cv2
import numpy as np
import torch


def enumerate_training_pairs(root_dir):
    """Read the released BIT folder convention, omitting its first pair."""
    pairs = []
    for scene in sorted(Path(root_dir).iterdir()):
        if not scene.is_dir():
            continue
        for scene_pair in sorted(scene.iterdir()):
            if not scene_pair.is_dir():
                continue
            for folder in sorted(scene_pair.iterdir()):
                if not folder.is_dir():
                    continue
                files = sorted(p for p in folder.iterdir() if p.is_file())
                for index in range(3, len(files), 2):
                    if index + 1 >= len(files):
                        raise ValueError(f'Missing transform after {files[index]}')
                    pairs.append({
                        'image0_path': str(files[0]),
                        'image1_path': str(files[index]),
                        'affine_matrix_path': str(files[index + 1]),
                    })
    # Reproduce the released order without resetting the process-wide RNG.
    np.random.RandomState(42).shuffle(pairs)
    return pairs


def resized_shape(width, height, resize=None, df=None):
    if resize is not None:
        ratio = resize / max(height, width)
        width, height = int(round(width * ratio)), int(round(height * ratio))
    if df is not None:
        if df <= 0:
            raise ValueError('The image division factor must be positive.')
        width, height = width // df * df, height // df * df
    if width <= 0 or height <= 0:
        raise ValueError('Image resizing produced an empty image.')
    return width, height


def prepare_gray_image(image, resize=None, df=None, padding=False):
    """Resize uint8 gray, pad on bottom/right, then return [1,H,W] in [0,1]."""
    if image.ndim != 2:
        raise ValueError('prepare_gray_image expects a single-channel H x W image.')
    height, width = image.shape
    new_width, new_height = resized_shape(width, height, resize, df)
    image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
    scale = torch.tensor([width / new_width, height / new_height], dtype=torch.float32)
    mask = None
    if padding:
        side = max(new_height, new_width)
        padded = np.zeros((side, side), dtype=image.dtype)
        padded[:new_height, :new_width] = image
        mask = torch.zeros((side, side), dtype=torch.bool)
        mask[:new_height, :new_width] = True
        image = padded
    return torch.from_numpy(image).float().unsqueeze(0) / 255, mask, scale


def read_clean_gray(path, resize=None, df=None, padding=False):
    """Use the same BGR-to-gray conversion as DGIM, before any augmentation."""
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f'Cannot read image: {path}')
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return prepare_gray_image(gray, resize, df, padding)


def preprocessing_spec(resize, df, padding):
    """Serializable cache contract; geometry is shared with student inputs."""
    return {
        'color': 'opencv_bgr_to_gray_repeated_rgb',
        'resize_long_edge': resize,
        'division_factor': df,
        'interpolation': 'opencv_linear',
        'pad_bottom_right': bool(padding),
        'padding_value_before_normalization': 0,
        'scale': 'uint8_div_255',
        'mean': [0.485, 0.456, 0.406],
        'std': [0.229, 0.224, 0.225],
        'center_crop': False,
    }
