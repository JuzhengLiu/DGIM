This directory is the default location for the DGIM matching checkpoint.

Obtain `dgim_512.ckpt` or `dgim_256.ckpt` from the authors' links in the root README. These matching checkpoints are not bundled. If stored elsewhere, set `matcher_checkpoint` in `configs/paths.local.yml`.

The bundled `src/fast_neural_style/weights/epoch5.pth` belongs to the training-time style generator and cannot substitute for the matching checkpoint.
