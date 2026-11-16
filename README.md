# DGIM: Cascaded Dynamic Data Generation for Robust Cross-Modal Image Matching

本仓库基于 [LotrL/DGIM](https://github.com/LotrL/DGIM) 的公开代码，增加集中路径管理，便于在 Windows、本地 Linux 和 AutoDL 之间迁移、提交不同实验版本。保留原始论文引用及 Apache-2.0 LICENSE。

上传内容包括全部现有源码、59 张风格图片、风格网络 `epoch5.pth`、示例图和架构图。`BIT_training_dataset` 由使用者单独上传；本仓库未附带 DGIM 匹配器的 `dgim_512.ckpt` / `dgim_256.ckpt`，需从下方作者链接取得。

本次改动只经过源码静态审查、Python 语法和 JSON 格式检查，没有执行训练/推理、安装依赖或验证论文指标。路径可迁移不代表论文已经完整复现；原有训练实现问题见下文。

## 集中路径配置

所有项目资源位置统一在 [`configs/paths.json`](configs/paths.json) 管理。模型超参数继续放在 `configs/dgim_ds.py` 和 `configs/multi_modality_512.py`。

- 配置中的相对路径以 **仓库根目录（`train.py` 所在目录）** 为基准，与启动命令时的工作目录无关。
- `--paths`、`--checkpoint` 等命令行路径覆盖参数也使用相同规则。可以使用绝对路径。
- 支持 `~`、`$VAR`、`${VAR}`、`%VAR%`；使用环境变量前需要先定义它。
- Windows 路径建议使用 `D:/datasets/...`；JSON 中反斜杠需要写成 `\\`。
- 输入资源不存在时不会自动下载；创建输出目录只会在之后真正运行相应程序时发生。

推荐只在本机创建 `configs/paths.local.json`，写入需要改变的几项；该文件已被 Git 忽略，后续 `git pull` 不会覆盖它。例如：

```json
{
  "paths": {
    "train_data_root": "D:/datasets/BIT_training_dataset/BIT_multi-modal_datasets_homography_30_rotation",
    "matcher_checkpoint": "D:/models/dgim_512.ckpt",
    "log_dir": "D:/experiments/DGIM/training",
    "demo_output_dir": "D:/experiments/DGIM/matches"
  }
}
```

默认路径适配原本的并列目录结构：

```text
workspace/
  DGIM/                       # 本仓库，原本地文件夹名为 DGIM-main
    configs/paths.json
    src/
    sample/
  BIT_training_dataset/       # 单独上传，不进入 Git
    BIT_multi-modal_datasets_homography_30_rotation/
    BIT_multi-modal_datasets_homography_180_rotation/
```

配置优先级：先读取 `paths.json`，再使用 `--paths` 指定的 JSON 覆盖；未传 `--paths` 时可以用 `DGIM_PATHS_CONFIG` 环境变量选择配置。只有两者都未设置时，才自动叠加 `paths.local.json`。显式指定的配置不会再混入本机配置。覆盖文件只需包含 `{"paths": {...}}` 及要修改的键，未知键会报错，避免拼写错误被忽略。

| 配置键 | 用途 |
|---|---|
| `data_config` / `model_config` | 数据设置 / 模型设置 Python 配置文件 |
| `train_data_root` | 训练数据根，下一层应为 `scene01` 等场景目录 |
| `style_dir` / `style_checkpoint` | 59 张风格图目录 / 风格网络权重 |
| `matcher_checkpoint` | 两个 demo 使用的 DGIM 权重 |
| `pretrained_checkpoint` | 训练初始化权重；`null` 表示不加载；不是恢复优化器状态 |
| `resume_checkpoint` | 恢复 Lightning 训练状态的 checkpoint；默认 `null`，与初始化权重按需求择一使用 |
| `log_dir` | TensorBoard 日志和默认训练输出根目录 |
| `checkpoint_dir` | 可选的 checkpoint 输出根；非空时保存至 `checkpoint_dir/实验名/version_N/`；默认随日志保存至 `log_dir/实验名/version_N/checkpoints/` |
| `sample_dir` | 批量 demo 的图像目录，以及 GUI 选图初始目录 |
| `demo_output_dir` | 匹配结果 PNG 输出目录 |
| `train_pose_root` / `train_npz_root` / `train_list_path` / `train_intrinsic_path` | 原配置兼容字段，当前 BIT loader 不使用，默认 `null` |
| `val_data_root` / `val_pose_root` / `val_npz_root` / `val_list_path` / `val_intrinsic_path` | 验证数据兼容字段；当前发布版无验证入口，默认 `null` |
| `test_data_root` / `test_pose_root` / `test_npz_root` / `test_list_path` / `test_intrinsic_path` | 测试数据兼容字段；当前发布版无完整指标评测入口，默认 `null` |

训练入口在合并超参数后，将这里的 train/val/test 路径写入 YACS 配置；不要再在模型配置里另外维护机器路径。`pretrained_checkpoint` 与 `resume_checkpoint` 分别可被 `--ckpt_path` 与 Lightning 的 `--resume_from_checkpoint` 临时覆盖。

## 在 AutoDL 获取代码与数据

以下是之后使用时的操作说明，本次整理没有执行训练或环境安装。

```bash
cd /root/autodl-tmp
git clone https://github.com/JuzhengLiu/DGIM.git
cd DGIM
cp configs/paths.autodl.example.json configs/paths.local.json
```

将数据单独放入 `/root/autodl-tmp/BIT_training_dataset/`，或在本机配置里改 `train_data_root`。切换 ±180° 数据版本也只需要改这一个键。若没有 `00.PNG`、变换图像及对应的 3×3 单应矩阵文本，只有压缩包，仍需自行解压到正确层级。

把匹配器权重放进仓库的 `weights/`，或将 `matcher_checkpoint` 指向服务器上已有的文件。风格网络权重已经随仓库提供，二者用途不同。

当依赖环境与权重准备好、确认需要运行后，服务器上的无界面演示命令为：

```bash
python demo.py --no-show
```

也可以直接选择示例配置，并临时覆盖输入/输出路径：

```bash
python demo.py --paths configs/paths.autodl.example.json --input-dir /root/autodl-tmp/my_pairs --output-dir /root/autodl-tmp/my_matches --no-show
```

输入图片按文件名排序，每相邻两张组成一对，请命名为 `001_0.png`、`001_1.png`、`002_0.png`、`002_1.png` 等。程序过滤非图像文件，遇到奇数张图像会明确报错。结果写到 `demo_output_dir`，默认情况下还会显示窗口；服务器应使用 `--no-show`。`demo_tk.py` 保留桌面文件选择界面，需要显示器。

训练路径入口已接入统一配置，例如之后可使用 `python train.py --paths configs/paths.autodl.example.json --exp_name experiment_v1`。但开始训练前还需核准下面的原始实现问题和依赖版本；这个命令不是已验证的完整复现流程。

## 保存不同代码版本

本地在仓库目录提交并上传：

```bash
git switch -c experiment/v1
# 修改代码；机器路径放 paths.local.json
git add .
git commit -m "Describe experiment v1 changes"
git push -u origin experiment/v1
```

服务器切换到同一版本：

```bash
git fetch origin
git switch --track origin/experiment/v1
```

已在该分支时，后续使用 `git pull --ff-only` 更新。需要固定实验版本时记录 `git rev-parse HEAD`，也可用 Git tag；本机路径配置与外部数据不会随分支切换。服务器若已经修改跟踪文件，应先提交或妥善保存修改，不要强制覆盖。

`.gitignore` 排除 BIT 数据及其压缩包、`data/`、`datasets/`、`outputs/`、日志、checkpoint 输出目录、缓存和本机路径覆盖文件。现有风格图片、示例和风格权重仍受版本控制；不要把服务器训练输出存到受跟踪的源码或资源目录。

## 复现状态与原始实现注意点

路径迁移改动包含：统一输入/权重/输出位置、加载权重时使用 CPU 映射、补全 `MGDPT_IMG_RESIZE` / `MGDPT_IMG_PAD` / `MGDPT_DF` 配置声明、排序并检查演示输入、增加无显示器输出与失败提示。模型结构、损失和训练阶段策略沿用原代码。

当前尚未解决、也未运行验证的上游问题包括：

- `src/dgim/utils/fine_matching.py` 的 `get_fine_match` 被 `torch.no_grad()` 包围，亚像素 MLP 的训练梯度需要核查。
- `src/fast_neural_style/transfer_net.py` 中第二组残差块返回值未赋回特征；修复前需要确认作者权重对应的代码版本。
- `train.py` 的预训练日程与 Dataset 当前启用的微调增强并不构成完整自动两阶段流程。
- 当前无完整论文指标评测入口；demo 的 MAGSAC 可视化不等于论文定量评测。
- 作者未提供依赖锁定文件。训练使用旧版 PyTorch Lightning API，不能直接认为最新版兼容；演示仍使用 CUDA 与半精度。
- 训练中的 `cv2.ximgproc.guidedFilter` 需要包含 contrib 模块的 OpenCV。其他依赖可从源码导入查看，尚未验证版本组合。

论文报告匹配器训练使用 RTX 4090D 24 GB；这不是本仓库已测得的最低要求。请在明确准备复现后再选择环境、核对数据版本并验证上述问题。

Paper Link: https://ieeexplore.ieee.org/document/11213501

![image](assets/architecture.png)

<br>

## 💾 Model Weights

Google Drive: https://drive.google.com/drive/folders/1bpHc09QDIr3idgPMNupS30NeLzGuqjh-?usp=drive_link

Baidu Netdisk: https://pan.baidu.com/s/14quWOrtOUOjgzJkAUzBsew?pwd=dgim

<br>

## 📦 Datasets Release

*** Training database ***

Google Drive: https://drive.google.com/drive/folders/1sMWudiDA2u7STSZuzm0Pg0IxtzNyYkeq?usp=sharing

Baidu Netdisk: https://pan.baidu.com/s/1eZ3wjIBBCAs0FgGgvpniMA?pwd=bit6

<br>

*** Testing dataset ***

Google Drive: https://drive.google.com/file/d/1cL0vBICIOQEPPJcJe7X89U4nRZWifRLU/view?usp=drive_link

Baidu Netdisk: https://pan.baidu.com/s/13u5VHJXNHewwRtT8EburRA?pwd=dgim

<br>

## 📚 Citation
```bibtex
@article{weng2025dgim,
  title={{DGIM}: Cascaded Dynamic Data Generation for Robust Cross-Modal Image Matching},
  author={Weng, Desheng and Li, Wei and Gao, Chenzhong and Xia, Xiang-Gen and Shi, Zhicheng and Cui, Bolun},
  journal={IEEE Transactions on Geoscience and Remote Sensing},
  year={2025},
  volume={63},
  pages={1-16},
  publisher={IEEE}
}
```
