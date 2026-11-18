# Local weights

All weights in this directory stay local and are excluded from Git. Paths are
managed in `configs/paths.yml`, with machine overrides in `paths.local.yml`.

| Resource | Use |
|---|---|
| `dinov3_convnext_large_pretrain_lvd1689m-61fa432d.pth` | Required only for offline teacher-cache preparation; obtain the official backbone state dictionary through https://github.com/facebookresearch/dinov3 |
| Original `dgim_512.ckpt` / `dgim_256.ckpt` | Optional author-baseline inference; incompatible with the custom RepViT backbone |
| `repvit_m1_5_distill_450e.pth` | v3 local reference resource only; **not loaded** by the custom student |

The new student and matching backend start from random initialization. After
training, checkpoints are written under `outputs/checkpoints/<experiment>/<stage>/<run>/` and include
the backbone type/configuration. A teacher checkpoint is not a matching model.

The already bundled `src/fast_neural_style/weights/epoch5.pth` is the frozen
style generator used during fine-tuning; it remains tracked as in the original
repository. DINOv3 source and license information is in `third_party/dinov3/`.
