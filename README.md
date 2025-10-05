# Branded Image Classification (PyTorch, end-to-end pipeline)

A lightweight, practical 3-class image classifier for e-commerce content quality and brand monitoring:
- Classes: `0` ordinary product photos (no logo), `1` brand logos, `2` product photos with watermarks
- Pipeline: data acquisition → cleaning/dedup → training → inference (supports URL and local paths)
- Deployment: CPU-friendly TorchScript for inference, with optional CUDA acceleration for training


## About the Author
- Tobias Chen — Python/Django developer focused on scalable, maintainable production backends and LLM APP Develop.
- Skills: Python, Django, REST APIs, Docker, PostgreSQL; ML/data governance; familiar with Celery/Redis and CI/CD.
- Interests: Web apps + image processing + automation + LLM APP Develop.
- Contact:
  - GitHub: https://github.com/TobiasChen-ML/
  - Email: tobiaschannel1999@gmail.com
  - Related project: Django-Vectorizer-Web (bitmap to SVG web app): https://github.com/TobiasChen-ML/Django-Vectorizer-Web


## Project Overview
- Goal: automatically detect whether an e-commerce image contains a brand logo or a watermark, to assist moderation and compliance.
- Models: default `resnet18` (small and fast); also supports `resnet34` and `vgg16`.
- Training: early stopping, class-imbalance mitigation (class-weighted loss/weighted sampler), and scheduled learning rate.
- Inference: CPU TorchScript deployment with CLI input via `--image`, `--folder`, or `--url` (http/https/file://).
- Data augmentation: script to synthesize “product + watermark” images from ordinary product photos to strengthen class 2.


## Project Structure
```
branded_image_classification/
├── artifacts/                 # Training artifacts (best weights and metadata)
│   └── resnet18_best/
│       ├── best.pt            # state_dict
│       ├── best_ts.pt         # TorchScript (CPU-friendly)
│       └── metadata.json      # model name, class names, image size, etc.
├── samples/                   # Synthetic sample data (optional)
├── src/
│   ├── download_data.py       # Crawl training data for all 3 classes (Bing/Google/Baidu)
│   ├── download_class2.py     # Targeted crawl for class 2 (product + watermark)
│   ├── clean_data.py          # Cleaning & dedup (size filter / corruption check / perceptual hash)
│   ├── prepare_sample_data.py # Generate simple synthetic data to validate the pipeline
│   ├── augment_watermark.py   # Synthesize “product + watermark” from ordinary product photos
│   ├── dataset.py             # DataLoader construction and train/val splitting
│   ├── models.py              # Model factory
│   ├── train.py               # Training entry
│   └── infer.py               # Inference entry (supports URL)
└── requirements.txt
```


## Quick Start
1) Install dependencies (use a virtual environment if possible):
```
pip install -r requirements.txt
```

2) Prepare data (choose one or combine):
- Generate simple synthetic data (for pipeline validation):
```
python src/prepare_sample_data.py --out-dir data --per-class 100 --img-size 256
```
- Crawl real data from the web (keywords per class):
```
python src/download_data.py --out-dir data --per-class 400 --per-keyword-limit 150 --engine bing
python src/clean_data.py --data-dir data --dedup --min-width 128 --min-height 128
```
  Custom keywords example:
```
python src/download_data.py --out-dir data --per-class 400 \
  --cls0 "street photo" "nature landscape" \
  --cls1 "brand logo vector" "company logo icon" \
  --cls2 "image watermark" "stock photo watermark"
```
- Specifically enrich class 2 (product + watermark):
```
python src/download_class2.py --out-dir data --count 800 --per-keyword-limit 200 --engines bing google baidu \
  --keywords "watermarked product photo" "brand watermark product" "商品 图片 水印"
```
- Synthesize class 2 from class 0 (fast augmentation):
```
python src/augment_watermark.py --src-dir data/train/0 --out-dir data/train/2 --count 500 --min-width 220 --min-height 220
```

3) Train (example: 50 epochs, ResNet18):
```
python src/train.py --data-dir data --model resnet18 --epochs 50 --batch-size 64
```
Common options:
- `--model {resnet18,resnet34,vgg16}`
- `--feature-extract` to train only the classifier head (faster, lower memory)
- `--img-size 224` can be adjusted (e.g., 192/160) to reduce memory, with possible accuracy trade-offs
- Class imbalance: `--class-weighted-loss`, `--weighted-sampler`
- Early stopping: `--es-metric {val_acc,val_loss}`, `--es-patience`, `--es-min-delta`
- On Windows, if `num_workers>0` causes an issue, set `--num-workers 0`

4) Inference (CPU, URL supported):
- Single local path:
```
python src/infer.py --model-artifacts artifacts/resnet18_best --image E:\path\to\image.jpg
```
- Single URL (HTTP/HTTPS/file://):
```
python src/infer.py --model-artifacts artifacts/resnet18_best --url "https://example.com/image.jpg"
python src/infer.py --model-artifacts artifacts/resnet18_best --url file:///E:/MyCode/branded_image_classification/samples/train/2/00000.jpg
```
- Batch folder:
```
python src/infer.py --model-artifacts artifacts/resnet18_best --folder E:\path\to\images
```
Notes: the inference script reads `metadata.json` to restore model structure and class names. TorchScript is enabled by default (`--use-torchscript`).


## FAQ / Notes
- CUDA acceleration: the project supports CPU inference; training can leverage CUDA-enabled `torch/torchvision/torchaudio` when available. Under restricted network conditions, CPU-only setups can still validate the pipeline.
- Windows paths and URLs: use `file:///E:/...` for Windows file URLs; examples above are compatible.
- Data quality: always run cleaning and dedup (`clean_data.py`) on crawled data; prefer real e-commerce images for stronger generalization.


## License
- This project follows the `LICENSE` in the repository.