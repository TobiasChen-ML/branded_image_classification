import os
import argparse
from typing import Tuple

from PIL import Image
import imagehash


def is_valid_image(path: str, min_size: Tuple[int, int]) -> bool:
    try:
        with Image.open(path) as img:
            img.verify()
        with Image.open(path) as img:
            w, h = img.size
            if w < min_size[0] or h < min_size[1]:
                return False
        return True
    except Exception:
        return False


def clean_folder(root: str, min_w: int, min_h: int, dedup: bool):
    seen_hashes = set()
    removed = 0
    checked = 0
    for dirpath, _, filenames in os.walk(root):
        for fn in filenames:
            p = os.path.join(dirpath, fn)
            checked += 1
            if not is_valid_image(p, (min_w, min_h)):
                try:
                    os.remove(p)
                    removed += 1
                except Exception:
                    pass
                continue
            if dedup:
                try:
                    with Image.open(p) as img:
                        ph = imagehash.phash(img)
                    if ph in seen_hashes:
                        os.remove(p)
                        removed += 1
                    else:
                        seen_hashes.add(ph)
                except Exception:
                    try:
                        os.remove(p)
                        removed += 1
                    except Exception:
                        pass
    print(f"Checked {checked} files, removed {removed} invalid/duplicate images.")


def parse_args():
    ap = argparse.ArgumentParser("Clean dataset: remove corrupted, too small, and duplicates")
    ap.add_argument("--data-dir", type=str, default="data")
    ap.add_argument("--min-width", type=int, default=128)
    ap.add_argument("--min-height", type=int, default=128)
    ap.add_argument("--dedup", action="store_true")
    return ap.parse_args()


def main():
    args = parse_args()
    train_root = os.path.join(args.data_dir, "train")
    val_root = os.path.join(args.data_dir, "val")
    for root in [train_root, val_root, args.data_dir]:
        if os.path.isdir(root):
            print(f"Cleaning {root} ...")
            clean_folder(root, args.min_width, args.min_height, args.dedup)


if __name__ == "__main__":
    main()