This directory is the default location for the DGIM matching checkpoint.

The matching checkpoints are not included in the local source supplied for this repository. Obtain `dgim_512.ckpt` or `dgim_256.ckpt` from the authors' Model Weights links in the root README, then set `matcher_checkpoint` in `configs/paths.local.json` if using a different location.

The bundled `src/fast_neural_style/weights/epoch5.pth` belongs to the training-time style generator and cannot substitute for the matching checkpoint.
