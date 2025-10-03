import os
import argparse
import random
from PIL import Image, ImageDraw, ImageFont


def make_bg(w, h, kind):
    img = Image.new("RGB", (w, h), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    if kind == 0:
        # 非品牌：简单几何形状
        for _ in range(5):
            x0, y0 = random.randint(0, w//2), random.randint(0, h//2)
            x1, y1 = random.randint(x0+10, w), random.randint(y0+10, h)
            color = tuple(random.randint(80, 200) for _ in range(3))
            draw.rectangle([x0, y0, x1, y1], outline=color, width=2)
    elif kind == 1:
        # logo：居中大文本或图形
        draw.ellipse([w*0.25, h*0.25, w*0.75, h*0.75], outline=(20, 20, 20), width=4)
        draw.text((w*0.35, h*0.45), "LOGO", fill=(30, 30, 30))
    else:
        # 水印：重复浅色文本
        for y in range(0, h, 40):
            for x in range(0, w, 80):
                draw.text((x, y), "WATERMARK", fill=(200, 200, 200))
    return img


def save_class_images(root, cls, per_class, size):
    out_dir = os.path.join(root, "train", str(cls))
    os.makedirs(out_dir, exist_ok=True)
    for i in range(per_class):
        img = make_bg(size, size, cls)
        img.save(os.path.join(out_dir, f"{i:05d}.jpg"))


def main():
    ap = argparse.ArgumentParser("Generate synthetic sample data for 3-class classification")
    ap.add_argument("--out-dir", type=str, default="data")
    ap.add_argument("--per-class", type=int, default=100)
    ap.add_argument("--img-size", type=int, default=256)
    args = ap.parse_args()

    for cls in [0, 1, 2]:
        save_class_images(args.out_dir, cls, args.per_class, args.img_size)

    print(f"Synthetic training data created under {args.out_dir}/train/{0,1,2}.")


if __name__ == "__main__":
    main()