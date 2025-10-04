import os
import argparse
import random
from typing import List, Tuple

from PIL import Image, ImageDraw, ImageFont


def list_images(root: str) -> List[str]:
    files = []
    for fn in os.listdir(root):
        p = os.path.join(root, fn)
        if os.path.isfile(p) and fn.lower().endswith((".jpg", ".jpeg", ".png")):
            files.append(p)
    return files


def is_large_enough(path: str, min_size: Tuple[int, int]) -> bool:
    try:
        with Image.open(path) as img:
            w, h = img.size
        return w >= min_size[0] and h >= min_size[1]
    except Exception:
        return False


def apply_watermark(src_img: Image.Image, text: str = "WATERMARK") -> Image.Image:
    img = src_img.convert("RGBA")
    w, h = img.size

    # Create a larger overlay and rotate it for diversity
    ov_w, ov_h = int(w * 1.5), int(h * 1.5)
    base_overlay = Image.new("RGBA", (ov_w, ov_h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(base_overlay)

    # Try loading a TrueType font from Windows; fallback to default
    font = None
    for fp in [
        r"C:\\Windows\\Fonts\\arial.ttf",
        r"C:\\Windows\\Fonts\\msyh.ttc",
        r"C:\\Windows\\Fonts\\song.ttf",
    ]:
        try:
            font = ImageFont.truetype(fp, size=random.randint(24, max(32, min(w, h) // 6)))
            break
        except Exception:
            continue
    if font is None:
        font = ImageFont.load_default()

    # Random tiling parameters
    step_x = random.randint(max(60, w // 10), max(120, w // 6))
    step_y = random.randint(max(40, h // 12), max(100, h // 7))
    alpha = random.randint(50, 140)
    color = random.choice([(255, 255, 255, alpha), (240, 240, 240, alpha), (200, 200, 200, alpha)])

    # Random jitter for positions
    jitter = 10
    for y in range(0, ov_h, step_y):
        for x in range(0, ov_w, step_x):
            jx = x + random.randint(-jitter, jitter)
            jy = y + random.randint(-jitter, jitter)
            draw.text((jx, jy), text, fill=color, font=font)

    # Optional grid lines
    if random.random() < 0.4:
        grid_alpha = max(20, alpha // 3)
        grid_color = (255, 255, 255, grid_alpha)
        for y in range(0, ov_h, step_y):
            draw.line([(0, y), (ov_w, y)], fill=grid_color, width=1)
        for x in range(0, ov_w, step_x):
            draw.line([(x, 0), (x, ov_h)], fill=grid_color, width=1)

    # Rotate overlay and crop center to image size
    angle = random.randint(-25, 25)
    rotated = base_overlay.rotate(angle, expand=True)
    rx, ry = rotated.size
    cx, cy = rx // 2, ry // 2
    left = max(0, cx - w // 2)
    top = max(0, cy - h // 2)
    crop = rotated.crop((left, top, left + w, top + h))

    result = Image.alpha_composite(img, crop).convert("RGB")
    return result


def parse_args():
    ap = argparse.ArgumentParser("Synthesize watermark product images from class-0 product photos")
    ap.add_argument("--src-dir", type=str, default=os.path.join("data", "train", "0"))
    ap.add_argument("--out-dir", type=str, default=os.path.join("data", "train", "2"))
    ap.add_argument("--count", type=int, default=500)
    ap.add_argument("--min-width", type=int, default=200)
    ap.add_argument("--min-height", type=int, default=200)
    return ap.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    src_files = [p for p in list_images(args.src_dir) if is_large_enough(p, (args.min_width, args.min_height))]
    if not src_files:
        raise RuntimeError(f"no valid source images found in {args.src_dir}")

    texts = ["WATERMARK", "BRAND", "COPYRIGHT", "DEMO", "SAMPLE", "品牌水印", "示例水印"]

    for i in range(args.count):
        src_path = random.choice(src_files)
        try:
            with Image.open(src_path) as img:
                wm_text = random.choice(texts)
                out_img = apply_watermark(img, wm_text)
                out_name = f"wmgen_{i:06d}.jpg"
                out_img.save(os.path.join(args.out_dir, out_name), quality=90)
        except Exception:
            continue

    print(f"Generated {args.count} watermarked product images into {args.out_dir}")


if __name__ == "__main__":
    main()