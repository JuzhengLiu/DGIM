# RepViT 官方资源

来源：[THU-MIG/RepViT](https://github.com/THU-MIG/RepViT)，CVPR 2024。保留上游 Apache-2.0 LICENSE。来源文件 Git blob 标识及本地权重校验值见 SOURCE.yml。

本目录保存官方骨干文件 model/repvit.py、model/__init__.py、utils.py 中的融合工具，以及上游 README 和依赖清单。没有修改模型逻辑；不是整个上游训练、检测、分割或 SAM 项目的镜像。上游说明中的相对链接可能需要在官方仓库查看。

资源路径统一登记在 configs/paths.yml 的 repvit_source_dir 和 repvit_checkpoint；机器差异写入 configs/paths.local.yml。M1.5 的 450-epoch 蒸馏权重已下载到本机，文件约 56.68 MiB；权重不提交到 GitHub，服务器需单独获取。

当前只是资源准备。DGIM 的 build_backbone 尚未接入 RepViT，原骨干仍是 RepVGG；DINOv3-L layer=23 的离线注意图和训练辅助损失也尚未实现。没有运行模型或安装 requirements.upstream.txt 中的依赖。

后续接入须保留完整空间特征图，适配 DGIM 所需的 /2、/4、/8 输出；不能直接调用原分类 forward 当作骨干。官方工厂函数的 pretrained 参数本身不负责加载这里的本地权重，需显式读取并核对骨干键、分类/蒸馏头和训练/融合结构，不能把此文件当作已训练的 DGIM 匹配器。上游 model/__init__.py 使用其原项目的绝对导入，本次仅原样保存，尚未提供本仓库的导入适配。
