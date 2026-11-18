import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import os
import re
import cv2
import random
import numpy as np
from pathlib import Path

from src.config.paths import load_paths, resolve_path
from src.datasets.preprocessing import enumerate_training_pairs, prepare_gray_image, read_clean_gray
from src.teacher.cache import TeacherCache


class MultiModality(Dataset):
    def __init__(self,
                 root_dir,
                 mode='train',
                 img_resize=None,
                 df=None,
                 img_padding=False,
                 style_dir=None,
                 style_checkpoint=None,
                 style_device='cpu',
                 training_stage='pretrain',
                 teacher_cache_dir=None,
                 teacher_enabled=False,
                 coarse_scale=0.125):
        """
        Read BIT image pairs and apply the selected DGIM training stage.

        Args:
            root_dir (str): BIT root directory containing scene subdirectories.
            mode (str): options are ['train', 'val', 'test'].
            img_resize (int, optional): the longer edge of resized images. None for no resize.
            df (int, optional): image size division factor. NOTE: this will change the final image size after img_resize.
            img_padding (bool): If set to 'True', zero-pad the image to squared size. This is useful during training.
            style_device: training device used by the frozen generator during finetuning.
        """
        super().__init__()
        self.root_dir = resolve_path(os.fspath(root_dir))
        if not Path(self.root_dir).is_dir():
            raise FileNotFoundError(f'Training data directory not found: {self.root_dir}. Update paths.train_data_root.')
        self.mode = mode
        if training_stage not in {'pretrain', 'finetune'}:
            raise ValueError("training_stage must be 'pretrain' or 'finetune'.")
        self.training_stage = training_stage
        self.dataset = []
        self.build_dataset()

        # parameters for image resizing and padding
        self.img_resize = img_resize
        self.df = df
        self.img_padding = img_padding

        # for training DGIM
        self.coarse_scale = coarse_scale

        # Pretraining is clean and never constructs or loads the style generator.
        self.style_model = None
        if self.mode == 'train' and self.training_stage == 'finetune':
            paths = load_paths() if style_dir is None or style_checkpoint is None else None
            self.style_folder_path = resolve_path(os.fspath(style_dir)) if style_dir is not None else paths['style_dir']
            if not Path(self.style_folder_path).is_dir():
                raise FileNotFoundError(f'Style directory not found: {self.style_folder_path}. Update paths.style_dir.')
            self.style_path_list = self.build_style_path_list()
            self.device = torch.device(style_device)
            self.style_model_path = resolve_path(os.fspath(style_checkpoint)) if style_checkpoint is not None else paths['style_checkpoint']
            if not Path(self.style_model_path).is_file():
                raise FileNotFoundError(f'Style checkpoint not found: {self.style_model_path}. Update paths.style_checkpoint.')
            self.style_model = self.build_style_model()

        self.teacher_cache_dir = resolve_path(os.fspath(teacher_cache_dir)) if teacher_cache_dir is not None else None
        self.teacher_cache = None
        self.teacher_enabled = teacher_enabled

    def build_dataset(self):
        self.dataset = enumerate_training_pairs(self.root_dir)

    @property
    def teacher_enabled(self):
        return self._teacher_enabled

    @teacher_enabled.setter
    def teacher_enabled(self, enabled):
        if enabled:
            if self.training_stage != 'pretrain' or self.mode != 'train':
                raise ValueError('Clean teacher targets are only used during pretraining.')
            if self.teacher_cache_dir is None:
                raise ValueError('teacher_cache_dir is required when teacher supervision is enabled.')
            if self.teacher_cache is None:
                self.teacher_cache = TeacherCache(
                    self.teacher_cache_dir, self.root_dir, self.img_resize, self.df, self.img_padding)
        self._teacher_enabled = bool(enabled)

    def set_teacher_enabled(self, enabled):
        """Call before creating each epoch's DataLoader and worker processes."""
        self.teacher_enabled = enabled

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]

        stylizing_option0 = stylizing_option1 = False
        thermal_option0 = thermal_option1 = False
        if self.mode == 'train' and self.training_stage == 'finetune':
            stylizing_option0 = random.choice([0, 1]) == 0
            stylizing_option1 = not stylizing_option0
            thermal_option0 = random.choice([0, 1]) == 0
            thermal_option1 = not thermal_option0

        # noinspection PyTypeChecker
        image0, mask0, scale0 = self.read_megadepth_gray(
            item["image0_path"], self.img_resize, self.df, self.img_padding, stylizing_option0, thermal_option0)

        # noinspection PyTypeChecker
        image1, mask1, scale1 = self.read_megadepth_gray(
            item["image1_path"], self.img_resize, self.df, self.img_padding, stylizing_option1, thermal_option1)

        # read and compute relative poses
        # noinspection PyTypeChecker
        T_0to1 = torch.from_numpy(np.loadtxt(item["affine_matrix_path"]).astype(np.float32))  # (3, 3)
        T_1to0 = T_0to1.inverse()

        data = {
            'image0': image0,  # (1, h, w)
            'image1': image1,
            'T_0to1': T_0to1,  # (3, 3)
            'T_1to0': T_1to0,
            'scale0': scale0,  # [scale_w, scale_h]
            'scale1': scale1,
            'dataset_name': 'Multi-Modality',
            'pair_id': idx,
        }

        # for DGIM training
        if mask0 is not None:  # Each view may have its own padded canvas.
            for key, mask in (('mask0', mask0), ('mask1', mask1)):
                data[key] = F.interpolate(mask[None, None].float(), scale_factor=self.coarse_scale,
                                          mode='nearest', recompute_scale_factor=False)[0, 0].bool()

        if self.teacher_enabled:
            for suffix, image, image_mask in (('0', image0, mask0), ('1', image1, mask1)):
                feature, valid = self.teacher_cache.load(item[f'image{suffix}_path'], image, image_mask)
                data[f'teacher_feature{suffix}'] = feature
                data[f'teacher_mask{suffix}'] = valid

        return data

    def build_style_path_list(self):
        style_path_list = []
        style_names = sorted(os.listdir(self.style_folder_path))
        for style_name in style_names:
            style_path = os.path.join(self.style_folder_path, style_name)
            style_path_list.append(style_path)
        return style_path_list

    def build_style_model(self):
        from src.fast_neural_style.transfer_net import TransferNet

        style_model = TransferNet()
        state_dict = torch.load(self.style_model_path, map_location='cpu')
        # remove saved deprecated running_* keys in InstanceNorm from the checkpoint
        for k in list(state_dict.keys()):
            if re.search(r"in\d+\.running_(mean|var)$", k):
                del state_dict[k]
        style_model.load_state_dict(state_dict)
        style_model.to(self.device).eval()
        # Freeze model parameters
        for param in style_model.parameters():
            param.requires_grad = False
        return style_model

    def stylize(self, content_path, style_path):
        from torchvision import transforms
        from src.fast_neural_style import utils

        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.mul(255))
        ])

        content_image = utils.load_image(content_path)
        content_image = transform(content_image)
        content_image = content_image.unsqueeze(0).to(self.device)

        style_image = utils.load_image(style_path)
        style_image = style_image.resize((256, 256))
        style_image = transform(style_image)
        style_image = style_image.unsqueeze(0).to(self.device)

        with torch.no_grad():
            output = self.style_model(content_image, style_image)
            # Two stride-2 encoder layers and 2x decoder stages round each
            # side up to a multiple of four. Keep the guide's pixel grid.
            height, width = content_image.shape[-2:]
            output = output[..., :height, :width]
            output = output.squeeze(0).clamp(0, 255).cpu().numpy().transpose(1, 2, 0).astype("uint8")

        # thermal_transform = BGRtoThermal()
        # guide_image = thermal_transform.augment_pseudo_thermal(cv2.imread(content_path))  # thermal image
        guide_image = cv2.imread(content_path)  # original image

        output_image = cv2.ximgproc.guidedFilter(guide=guide_image,
                                                 src=cv2.cvtColor(output, cv2.COLOR_RGB2BGR),
                                                 radius=15, eps=0.001)
        # output_image = cv2.cvtColor(output_image, cv2.COLOR_BGR2GRAY)  # stylized image
        # output_image = cv2.cvtColor(guide_image, cv2.COLOR_BGR2GRAY)  # thermal or original image
        return output_image

    def imread_gray(self, path, stylizing_option, thermal_option):
        if stylizing_option:
            image = self.stylize(path, random.choice(self.style_path_list))
        else:
            image = cv2.imread(path)
            if image is None:
                raise FileNotFoundError(f'Cannot read image: {path}')

        if thermal_option:
            thermal_transform = BGRtoThermal()
            image = thermal_transform.augment_pseudo_thermal(image)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        return image  # (h, w)

    def read_megadepth_gray(self, path, resize=None, df=None, padding=False, stylizing_option=False, thermal_option=False):
        """
        Args:
            resize (int, optional): the longer edge of resized images. None for no resize.
            padding (bool): If set to 'True', zero-pad resized images to squared size.
        Returns:
            image (torch.tensor): (1, h, w)
            mask (torch.tensor): (h, w)
            scale (torch.tensor): [w/w_new, h/h_new]
        """
        if not stylizing_option and not thermal_option:
            return read_clean_gray(path, resize, df, padding)
        image = self.imread_gray(path, stylizing_option, thermal_option)
        return prepare_gray_image(image, resize, df, padding)


class BGRtoThermal:
    def __init__(self):
        os.environ.setdefault('NO_ALBUMENTATIONS_UPDATE', '1')
        import albumentations as A

        self.blur = A.Blur(p=0.7, blur_limit=(3, 5))  # default: blur_limit=(2, 4)
        self.hsv = A.HueSaturationValue(p=0.9, val_shift_limit=(-30, +30), hue_shift_limit=(-90, +90),
                                        sat_shift_limit=(-30, +30))

        # parameters for the cosine transform
        self.w_0 = np.pi * 2 / 3
        self.w_r = np.pi / 2
        self.theta_r = np.pi / 2

    def augment_pseudo_thermal(self, image):
        # HSV augmentation
        image = self.hsv(image=image)["image"]

        # Random blur
        image = self.blur(image=image)["image"]

        # Convert the image to the gray scale
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Normalize the image between (-0.5, 0.5)
        image = image / 255 - 0.5

        # Random phase and freq for the cosine transform
        phase = np.pi / 2 + np.random.randn(1) * self.theta_r
        w = self.w_0 + np.abs(np.random.randn(1)) * self.w_r

        # Cosine transform
        image = np.cos(image * w + phase)

        # A constant image has no contrast; keep it zero instead of dividing by zero.
        image = (image - image.min()) / max(image.max() - image.min(), 1e-8) * 255

        # 3 channel gray
        # image = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_GRAY2BGR)

        return image.astype(np.uint8)
