This directory is the default location for the DGIM matching checkpoint.

Obtain `dgim_512.ckpt` or `dgim_256.ckpt` from the authors' links in the root README. These matching checkpoints are not bundled. If stored elsewhere, set `matcher_checkpoint` in `configs/paths.local.yml`.

The bundled `src/fast_neural_style/weights/epoch5.pth` belongs to the training-time style generator and cannot substitute for the matching checkpoint.

RepViT-M1.5 resource preparation (v3): `repvit_m1_5_distill_450e.pth` is the official ImageNet-1K classification checkpoint from THU-MIG/RepViT, trained for 450 epochs with distillation (reported Top-1: 82.5%). It is not a trained DGIM matcher or a DINOv3 checkpoint. It is stored locally only; upload it separately to your server, or obtain it from the upstream release:

https://github.com/THU-MIG/RepViT/releases/download/v1.0/repvit_m1_5_distill_450e.pth

Set `repvit_checkpoint` in `configs/paths.local.yml` if you use a different location. All files under this weights directory except this README are ignored by Git. Source and checkpoint provenance are recorded in `third_party/repvit/SOURCE.yml`.
