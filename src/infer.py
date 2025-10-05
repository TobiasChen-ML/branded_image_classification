import os
import json
import argparse
from typing import List
import tempfile
from urllib.parse import urlparse

import torch
from PIL import Image
from torchvision import transforms
import requests

import sys
CURR_DIR = os.path.dirname(os.path.abspath(__file__))
if CURR_DIR not in sys.path:
    sys.path.append(CURR_DIR)

from models import create_model


def build_val_transform(img_size: int):
    return transforms.Compose([
        transforms.Resize(int(img_size * 1.2)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def load_metadata(artifacts_dir: str):
    meta_path = os.path.join(artifacts_dir, "metadata.json")
    if not os.path.isfile(meta_path):
        raise FileNotFoundError(f"metadata.json not found in {artifacts_dir}")
    with open(meta_path, "r", encoding="utf-8") as f:
        return json.load(f)


def infer_image(model, tf, image_path: str, device: str, class_names: List[str]):
    img = Image.open(image_path).convert("RGB")
    x = tf(img).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)[0]
        pred = torch.argmax(probs).item()
    return class_names[pred], probs.cpu().tolist()


def parse_args():
    ap = argparse.ArgumentParser("CPU inference for 3-class branded image classifier")
    ap.add_argument("--model-artifacts", type=str, required=True, help="path to artifacts/<model>_best directory")
    ap.add_argument("--image", type=str, help="single image path")
    ap.add_argument("--folder", type=str, help="folder of images to batch infer")
    ap.add_argument("--url", type=str, help="single image URL (http/https/file)")
    ap.add_argument("--use-torchscript", action="store_true", default=True)
    ap.add_argument("--quantize", action="store_true", help="use dynamic quantization (not with torchscript)")
    ap.add_argument("--img-size", type=int, default=None, help="override image size if needed")
    ap.add_argument("--topk", type=int, default=1)
    return ap.parse_args()


def build_cpu_model(artifacts_dir: str, use_torchscript: bool, quantize: bool):
    meta = load_metadata(artifacts_dir)
    model_name = meta["model_name"]
    class_names = meta["class_names"]
    img_size = meta["img_size"]

    if use_torchscript:
        ts_path = os.path.join(artifacts_dir, "best_ts.pt")
        if not os.path.isfile(ts_path):
            raise FileNotFoundError("best_ts.pt not found")
        model = torch.jit.load(ts_path, map_location="cpu")
        model.eval()
    else:
        ckpt = torch.load(os.path.join(artifacts_dir, "best.pt"), map_location="cpu")
        model = create_model(model_name, num_classes=len(class_names), pretrained=False, feature_extract=meta.get("feature_extract", False))
        model.load_state_dict(ckpt["model_state"])
        model.eval()
        if quantize:
            model = torch.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)

    return model, class_names, img_size


def main():
    torch.set_num_threads(max(1, os.cpu_count() or 1))
    args = parse_args()

    model, class_names, img_size = build_cpu_model(args.model_artifacts, args.use_torchscript, args.quantize)
    tf = build_val_transform(args.img_size or img_size)

    device = "cpu"
    results = []

    if args.image:
        label, probs = infer_image(model, tf, args.image, device, class_names)
        results.append({"path": args.image, "label": label, "probs": probs})
    elif args.url:
        parsed = urlparse(args.url)
        if parsed.scheme in ("file", ""):
            # file URL or local path (Windows-compatible)
            if parsed.scheme == "file":
                if parsed.netloc:
                    # e.g., file://E:/path -> netloc='E:', path='/path'
                    local_path = (parsed.netloc + parsed.path).lstrip('/')
                else:
                    # e.g., file:///E:/path -> path='/E:/path'
                    local_path = parsed.path.lstrip('/')
                local_path = local_path.replace('/', os.sep)
            else:
                local_path = args.url
            label, probs = infer_image(model, tf, local_path, device, class_names)
            results.append({"path": args.url, "label": label, "probs": probs})
        elif parsed.scheme in ("http", "https"):
            try:
                with requests.get(args.url, timeout=15, stream=True) as r:
                    r.raise_for_status()
                    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
                        for chunk in r.iter_content(chunk_size=8192):
                            if chunk:
                                tmp.write(chunk)
                        temp_path = tmp.name
                label, probs = infer_image(model, tf, temp_path, device, class_names)
                results.append({"path": args.url, "label": label, "probs": probs})
                try:
                    os.remove(temp_path)
                except Exception:
                    pass
            except Exception as e:
                results.append({"path": args.url, "error": str(e)})
    elif args.folder:
        for name in os.listdir(args.folder):
            p = os.path.join(args.folder, name)
            if not os.path.isfile(p):
                continue
            try:
                label, probs = infer_image(model, tf, p, device, class_names)
                results.append({"path": p, "label": label, "probs": probs})
            except Exception as e:
                results.append({"path": p, "error": str(e)})
    else:
        raise ValueError("please provide --image or --folder or --url")

    for r in results:
        if "error" in r:
            print(f"{r['path']} -> ERROR: {r['error']}")
        else:
            print(f"{r['path']} -> {r['label']} | probs={r['probs']}")
            return r['label']

if __name__ == "__main__":
    main()