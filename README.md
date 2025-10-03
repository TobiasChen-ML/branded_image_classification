# 品牌图像三分类（PyTorch，小模型，CUDA训练/CPU推理）

本项目提供一个内存占用较小的三分类图像分类器：
- 类别：`0` 非品牌图、`1` logo图、`2` 水印图
- 训练：使用 CUDA（如可用）与混合精度以减少显存占用、提升速度
- 推理：在 CPU 上进行，支持 TorchScript 加速与可选动态量化
- 模型：默认 `resnet18`（小且快），也支持 `resnet34`、`vgg16`

目录结构建议（自备数据）
- `data/train/0`, `data/train/1`, `data/train/2`
- `data/val/0`, `data/val/1`, `data/val/2`（如无 `val`，训练脚本会自动从 `train` 切分）

快速开始
1) 创建合成示例数据（用于验证管线，非真实业务）：
```
python src/prepare_sample_data.py --out-dir data --per-class 100
```
2) 从互联网下载真实数据（可选）：
```
pip install -r requirements.txt
python src/download_data.py --out-dir data --per-class 300
python src/clean_data.py --data-dir data --dedup --min-width 128 --min-height 128
```
说明：`download_data.py` 默认会分别抓取 0/1/2 类的关键词图片到 `data/train/0|1|2`，你也可以传入自定义关键词：
```
python src/download_data.py --out-dir data --per-class 400 \
  --cls0 "street photo" "nature landscape" \
  --cls1 "brand logo vector" "company logo icon" \
  --cls2 "image watermark" "stock photo watermark"
```

3) 训练（默认 resnet18，小内存，快）：
```
python src/train.py --data-dir data --model resnet18 --epochs 10 --batch-size 64 --mixed-precision
```
常用参数：
- `--model {resnet18,resnet34,vgg16}`
- `--feature-extract` 仅训练分类头，速度更快、显存更低
- `--img-size 224` 可调小（如 192/160）进一步降内存，但可能降精度
- `--batch-size` 根据显存调节；开启混合精度可用更大 batch
- 类不平衡处理：
  - `--class-weighted-loss` 使用类别权重（按训练集频次反比）
  - `--weighted-sampler` 训练集使用加权采样（提升少数类采样概率）
- 早停（避免过拟合与节约时间）：
  - `--es-metric {val_acc,val_loss}` 监控指标（默认 `val_acc`）
  - `--es-patience 5` 容忍未提升的轮数
  - `--es-min-delta 1e-4` 最小提升幅度

4) 推理（CPU，快速）：
```
python src/infer.py --model-artifacts artifacts/resnet18_best --image path/to/test.jpg
```
或对文件夹批量：
```
python src/infer.py --model-artifacts artifacts/resnet18_best --folder path/to/images
```
可选：
- `--use-torchscript`（默认开启）使用 TorchScript 加速
- `--quantize` 对线性层做动态量化（不与 TorchScript 同时使用）

数据准备建议
- 使用 `ImageFolder` 结构，分别放置 0/1/2 三类图片
- 图片尺寸不限，脚本会自动缩放到 `--img-size`
- 如果暂时没有数据，可用 `prepare_sample_data.py` 生成简易合成数据先跑通流程

训练细节
- 优化器：AdamW，`lr=3e-4` 默认
- 学习率调度：StepLR（每 5 个 epoch 衰减 0.1）
- 损失：CrossEntropyLoss
- 增广：随机裁剪、水平翻转、轻微颜色抖动（可根据数据情况调整）
- 保存：`artifacts/<model>_best/` 下保存 `best.pt`（state_dict）、`best_ts.pt`（TorchScript）、`metadata.json`（模型/类别/尺寸等）

推理细节
- 读取 `metadata.json` 还原模型结构与类别名
- 默认为 CPU 推理；可选 TorchScript 加速或动态量化（线性层）

依赖安装
```
pip install -r requirements.txt
```

在 Windows 上的注意事项
- 如果 `DataLoader` 的 `num_workers>0` 遇到问题，可改为 `0`
- 请使用 `python` 直接运行脚本；确保 `src` 目录在同级

许可证
- 本项目遵循仓库中的 `LICENSE`