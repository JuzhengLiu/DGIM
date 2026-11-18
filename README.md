# DGIM：定制 RepViT 与训练期 DINOv3 引导

当前版本 **v6**。本版修正匹配监督、空候选处理和离线缓存配置。已完成静态审查，训练显存、速度和匹配精度仍待实测。[作者代码](https://github.com/LotrL/DGIM) · [版本记录](CHANGELOG.md) · [v5 静态审查记录](docs/AUDIT_v5.md)

预训练与微调是**两次手动启动的独立训练**，前者结束后程序退出，不自动启动后者：

```bash
python train.py --stage pretrain
```

第一阶段结束，确认要加载的权重后，再运行：

```bash
python train.py --stage finetune
```

## 当前模型

学生使用灰度、随机初始化的定制 RepViT，保留 DGIM 的自/交叉注意力、粗匹配、多尺度融合、5×5细匹配和亚像素模块。学生不加载官方 M1.5/M2.3 分类权重；原 RepVGG 仍可用于同条件基线。

| 部分 | 块数与步幅 | 512输入输出 |
|---|---|---|
| stem | 1→64，stride2；64→128，stride1 | 128×256×256 |
| Stage1 | 2个普通块 | 128×256×256，1/2 |
| Stage2 | 首个stride2过渡块＋3个普通块 | 196×128×128，1/4 |
| Stage3 | 首个stride2过渡块＋27个普通块 | 256×64×64，1/8 |
| Stage4 | 2个普通块，SE关/开，无新过渡层 | 256×64×64，1/8 |

Stage3/4的28＋2仅拆分连续的30块。卷积使用Kaiming初始化，通道残差末端BN保留零初始化。196是精确通道数，不执行上游的8倍数取整。估算部署骨干约9.42M参数、56.90 GMAC/512单图；这是静态算术，不是实测速度。

```text
原图 → DGIM灰度/尺寸预处理 → 定制RepViT → 原DGIM匹配后端 → 对应点
                 │               │
                 │               └→ 原始1/8特征 → 训练投影256→384 ┐
                 └→ 灰度复制3通道 → 冻结DINO → 离线1/8缓存 ──────┴→ cosine辅助损失
```

DINOv3 ConvNeXt-Large保持官方RGB接口与全部预训练参数，输入灰度复制三份并使用官方均值/标准差。离线提取原生第二stage（索引1，384通道），只计算前两个stage；没有ViT第23层、没有CLS注意力蒸馏。教师1/8格点中心8i+3.5通过固定双线性采样对齐学生8i，屏蔽插值越界与补边。教师不进入学生训练前向或最终推理模型。

总损失保留原DGIM的 `0.5 L_coarse + 0.3 L_fine + 0.1 L_sub`，另加有权重的教师特征项。按从1开始的轮次计数，第1–5轮教师权重为`0.01/0.02/0.03/0.04/0.05`，第6–10轮为`0.05/0.04/0.03/0.02/0.01`，第11轮起归零并停止读取缓存。初始最大权重0.05尚未经过实验调优。微调阶段始终关闭教师。预训练25轮含5轮学习率warmup；微调初始LR为1.25e-4，默认最多25轮，并按训练匹配损失早停。这个收敛判断不是论文未公开细节的保证性复现。

## 服务器获取与更新

首次克隆：

```bash
cd /root/autodl-tmp
git clone https://github.com/JuzhengLiu/DGIM.git
cd DGIM
cp configs/paths.autodl.example.yml configs/paths.local.yml
```

仅首次且没有本机配置时执行`cp`。以后每次更新始终使用：

```bash
cd /root/autodl-tmp/DGIM
git switch main
git pull --ff-only
```

需要固定 v6 做对照实验时，依次执行 `git fetch --tags` 和 `git switch --detach v6`。返回最新版用上面的三行。

## 路径和资源

所有资源位置由 [configs/paths.yml](configs/paths.yml) 管理；机器差异只写入不上传的`configs/paths.local.yml`。相对路径以仓库根目录为准，Windows可写`D:/datasets/...`。`--paths`或`DGIM_PATHS_CONFIG`选择其他YML覆盖默认值。

```yaml
paths:
  train_data_root: /root/autodl-tmp/BIT_training_dataset/BIT_multi-modal_datasets_homography_30_rotation
  teacher_checkpoint: /root/autodl-tmp/DGIM/weights/dinov3_convnext_large_pretrain_lvd1689m-61fa432d.pth
  teacher_cache_dir: /root/autodl-tmp/DGIM_outputs/teacher_cache
```

`train_data_root`下一层应为`scene01`等目录。数据自行上传，DINO权重从[官方项目](https://github.com/facebookresearch/dinov3)申请获取后单独上传；程序不自动下载。教师需要官方backbone的原始state_dict，不接收其他框架的任意包装checkpoint。

仓库保留原59张风格图、风格生成器`src/fast_neural_style/weights/epoch5.pth`、sample和assets。数据、DINO/RepViT/DGIM外部权重、新训练checkpoint、教师缓存和本机路径不上传。v3保存的M1.5权重在本版本不使用。

模型设置在`configs/dgim_ds.py`，输入尺寸在`configs/multi_modality_512.py`；两个阶段的训练参数分别在下面的YML中。改变输入预处理、教师权重或教师源码后应使用新的缓存目录重新提取。

## 两阶段配置与权重保存

| 内容 | 配置文件/字段 |
|---|---|
| 预训练轮数、学习率、batch、保存频率、教师权重 | [configs/pretrain.yml](configs/pretrain.yml) |
| 微调轮数、学习率、batch、保存频率、早停 | [configs/finetune.yml](configs/finetune.yml) |
| 第二阶段加载哪个权重 | `paths.finetune_init_checkpoint` |
| 第一阶段可选初始化权重 | `paths.pretrain_init_checkpoint`，默认null |
| 同阶段断点续训 | `paths.pretrain_resume_checkpoint` / `paths.finetune_resume_checkpoint` |
| 权重保存根目录 | `paths.checkpoint_dir` |
| 新实验子目录 | 对应阶段YML中的`TRAINER.RUN_NAME` |

两个阶段都可独立修改，例如：

```yaml
TRAINER:
  RUN_NAME: run01
  MAX_EPOCHS: 25
  BATCH_SIZE: 4
  LR: 0.000125
  SAVE_EVERY_N_EPOCHS: 5
  SAVE_LAST: true
  EARLY_STOP_ENABLED: true
  EARLY_STOP_PATIENCE: 5
  EARLY_STOP_MIN_DELTA: 0.0001
```

`SAVE_EVERY_N_EPOCHS: 5`表示完成第5、10、15…轮保存；正常末轮或早停轮即使不满5轮也保存。保存发生在完整epoch结束时，不做中途batch恢复。进程被强制中断时，仅能从最近已保存的轮次恢复。`MAX_EPOCHS`是本阶段总轮数上限；恢复训练时它不是额外新增轮数。早停监控`train/epoch_loss_match`，不把逐轮变化的教师损失混入早停判断；设`EARLY_STOP_ENABLED: false`即可固定跑满指定轮数。

默认保存位置：

```text
outputs/checkpoints/repvit_dino/
├── pretrain/run01/
│   ├── epoch=001-step=0001807.ckpt
│   ├── epoch=002-step=0003614.ckpt
│   └── last.ckpt
└── finetune/run01/
    ├── epoch=001-step=0001807.ckpt
    └── last.ckpt
```

轮数从1开始；步数由实际数据量、batch和设备数决定，上面仅为本地BIT数据、单GPU、batch4、无梯度累积的示例。历史归档一直保留，同名冲突会加版本后缀。**只有`last.ckpt`是有意更新的最新副本**，且只在上述保存事件更新；`SAVE_LAST: false`时只保存历史归档。第二阶段可以加载`last.ckpt`，也可以指定任何某轮历史归档。

新实验把`RUN_NAME`改为`run02`；如果已有checkpoint还用同一目录重新开始，程序会直接提示换名或显式resume，不覆盖旧实验。`--resume`继续写入所指定checkpoint的父目录。日志在`log_dir/实验名/阶段/运行名/version_*`，与权重路径分开；checkpoint内保存生效配置、初始化权重来源和运行目录。

第二阶段选择某个第一阶段权重，在`configs/paths.local.yml`中设置：

```yaml
paths:
  finetune_init_checkpoint: outputs/checkpoints/repvit_dino/pretrain/run01/epoch=010-step=0018070.ckpt
```

此处仅初始化匹配器，微调的优化器、学习率和轮数从本阶段重新开始。resume则恢复完整训练状态，并优先于YML中的init路径；不同时使用`--pretrained`和`--resume`。更改`checkpoint_dir`、实验名或`RUN_NAME`后，要同步选择实际的`finetune_init_checkpoint`和推理`matcher_checkpoint`，程序不猜测应该使用哪个实验。

命令行只覆盖明确传入的值，如`--epochs 50 --save-every-n-epochs 2 --run-name run02`。可以通过`--config`选择另一份阶段YML，或在路径YML中修改`pretrain_config`/`finetune_config`；模型/数据配置先合并，阶段YML随后合并，CLI最后覆盖。

从 v4 升级时，将本机 YML 中的 `pretrained_checkpoint` 改为 `finetune_init_checkpoint`，将 `resume_checkpoint` 改为对应阶段的 `pretrain_resume_checkpoint` 或 `finetune_resume_checkpoint`。v4 匹配器 checkpoint 可通过 `--pretrained` 初始化同结构模型。v5 和 v6 的完整续训使用相同的训练元数据格式。要复现 v5 的训练轨迹，请固定 v5 代码，v6 已修正监督和候选选择。

## 后续在服务器执行

下列命令是使用说明，本次没有执行。目标依赖组合为Python3.10、PyTorch2.2.2、Lightning2.2.5，见[requirements.txt](requirements.txt)，没有声称这套组合已安装或通过实测。未来准备环境时：

```bash
pip install -r requirements.txt
pip install --no-deps -r requirements-augmentation.txt
```

第二步保留原Albumentations1.3.1增强；其依赖已列在第一份清单。使用`--no-deps`是为了避免另装非contrib的OpenCV覆盖`cv2.ximgproc.guidedFilter`，不需要额外安装opencv-python或opencv-python-headless。

### 1. 生成一次教师缓存

```bash
python prepare_teacher_cache.py
```

默认读取 `configs/pretrain.yml`。若预训练指定了自定义配置，生成缓存时也传入相同的 `--config`、`--model-config` 或 `--data-config`。两者按模型、数据、阶段的顺序合并设置。教师来源、数据和预处理相同的已有缓存可继续使用。

它按训练集引用的去重原图提取干净灰度目标，保存FP16特征和有效mask，并写`manifest.yml`记录教师、预处理和完成状态。相同图片多次用于配对不会重复保存。缓存按数据根目录内的相对路径定位，便于迁移服务器。512输入下8030张图片的裸特征约25.26 GB，另有mask与文件元数据；实际图片数以数据集为准。这是使用教师路线的一次性独立准备命令，训练入口不会自动提取、下载或执行预检训练；无教师对照不需要这一步。

### 2. 预训练学生和匹配后端

```bash
python train.py --stage pretrain
```

默认实验名`repvit_dino`，batch4，单GPU；从随机参数开始。预训练关闭动态风格和余弦增强，不加载风格生成器。学习率warmup按实际优化步数计算，不再写死每轮1798步。

读取`configs/pretrain.yml`。默认输出：`outputs/checkpoints/repvit_dino/pretrain/run01/last.ckpt`。第一阶段结束即退出；`finetune_init_checkpoint`默认指向这里，供随后手动启动的第二阶段使用。

### 3. 微调

```bash
python train.py --stage finetune
```

读取`configs/finetune.yml`和`finetune_init_checkpoint`中的同结构匹配器参数；恢复原随机风格/余弦强度增强，冻结生成器。默认`NUM_WORKERS: 0`，使风格生成器留在训练主进程，与训练模型使用同一设备。输出：`outputs/checkpoints/repvit_dino/finetune/run01/last.ckpt`。

需要延长微调上限时用`--epochs 50`。早停耐心和阈值在配置中调整，不把训练loss收敛等同于验证集最优。

### 4. 恢复中断或运行对照

恢复同一阶段，保留优化器、学习率和步数：

```bash
python train.py --stage pretrain --resume outputs/checkpoints/repvit_dino/pretrain/run01/last.ckpt
```

只换骨干、不用教师：

```bash
python train.py --stage pretrain --no-teacher
python train.py --stage finetune --no-teacher --pretrained outputs/checkpoints/repvit/pretrain/run01/last.ckpt
```

原RepVGG同条件基线：

```bash
python train.py --stage pretrain --backbone repvgg --no-teacher
python train.py --stage finetune --backbone repvgg --no-teacher --pretrained outputs/checkpoints/repvgg/pretrain/run01/last.ckpt
```

这些对照不需要教师缓存。已发布作者权重只能用于原RepVGG，不能作为定制学生的严格整网初始化。不要把恢复训练与微调阶段的模型参数初始化混用。

### 5. 推理

```bash
python demo.py --no-show
```

从`matcher_checkpoint`读取模型配置及匹配器参数，转换为部署结构后推理，不加载教师和辅助头。默认路径指向新模型微调的`last.ckpt`。原作者checkpoint没有v4配置元数据，演示仍按原RepVGG结构加载，例如`python demo.py --checkpoint weights/dgim_512.ckpt --no-show`。`demo_tk.py`用于有桌面的机器。

## 与基线共用的修正及验证范围

- 离散细匹配选择继续不求导，亚像素MLP及训练坐标恢复梯度；不再原地改动用于fine loss的置信矩阵。重投影距离改用等价的vector_norm，避免零距离处sqrt导数问题。
- 预训练/微调切换、实际步数warmup、checkpoint恢复和新旧骨干演示使用同一套明确接口。没有在线教师、自动下载、自动改模型或错误后退回另一条训练路径。
- 复用原DGIM注意力、候选筛选和几何监督。原风格网络及其权重保持配套，不擅自改变其内部残差调用行为。
- v5修复半精度粗匹配掩码溢出、BF16 RoPE类型、半精度亚像素坐标/OpenCV输入、空GT分支、padding监督、增强零分母和Pillow10接口，并删除失效通用增强配置与逐图CUDA缓存清空，详见[静态审查记录](docs/AUDIT_v5.md)。
- v6 用显式边界判断保留粗网格原点和细窗口左上角的有效监督，细匹配全部低于阈值时返回空结果。亚像素损失在除法前排除投影分母为零的点。四个回归用例覆盖恒等变换、低置信度、无重叠和投影分母为零的情形。[回归用例](tests/test_matching_edges.py)
- 仅做源码交叉审查、Python语法和YML静态检查。未运行任何模型或测试，因此实际权重加载、融合数值等价、显存、训练稳定性、推理速度和准确率尚未实测。
- 原仓库没有完整论文指标评测入口；demo的MAGSAC可视化不是定量复现。需要同条件比较原DGIM、新骨干无教师、新骨干加教师，才能判断实验收益。

## 来源和许可

DGIM保留根目录Apache-2.0许可；RepViT参考源码及许可见[third_party/repvit](third_party/repvit)，定制学生实现见[src/dgim/backbone/repvit.py](src/dgim/backbone/repvit.py)。DINO教师派生源码单独遵循[DINOv3 License](third_party/dinov3/LICENSE.md)，来源见[third_party/dinov3/source.yml](third_party/dinov3/source.yml)，不将其重新标为Apache-2.0。

[DGIM论文](https://ieeexplore.ieee.org/document/11213501) · [作者代码](https://github.com/LotrL/DGIM) · [RepViT](https://github.com/THU-MIG/RepViT) · [DINOv3](https://github.com/facebookresearch/dinov3)

![Original DGIM architecture](assets/architecture.png)

| 原论文资源 | Google Drive | 百度网盘 |
|---|---|---|
| 模型权重 | [下载](https://drive.google.com/drive/folders/1bpHc09QDIr3idgPMNupS30NeLzGuqjh-?usp=sharing) | [提取码dgim](https://pan.baidu.com/s/14quWOrtOUOjgzJkAUzBsew?pwd=dgim) |
| 训练数据 | [下载](https://drive.google.com/drive/folders/1sMWudiDA2u7STSZuzm0Pg0IxtzNyYkeq?usp=sharing) | [提取码bit6](https://pan.baidu.com/s/1eZ3wjIBBCAs0FgGgvpniMA?pwd=bit6) |
| 测试数据 | [下载](https://drive.google.com/file/d/1cL0vBICIOQEPPJcJe7X89U4nRZWifRLU/view?usp=sharing) | [提取码dgim](https://pan.baidu.com/s/13u5VHJXNHewwRtT8EburRA?pwd=dgim) |

```bibtex
@article{weng2025dgim,
  title={{DGIM}: Cascaded Dynamic Data Generation for Robust Cross-Modal Image Matching},
  author={Weng, Desheng and Li, Wei and Gao, Chenzhong and Xia, Xiang-Gen and Shi, Zhicheng and Cui, Bolun},
  journal={IEEE Transactions on Geoscience and Remote Sensing},
  year={2025}, volume={63}, pages={1-16}, publisher={IEEE}
}
```
