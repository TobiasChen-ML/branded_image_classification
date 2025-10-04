import os
import argparse
from typing import List

from icrawler.builtin import BingImageCrawler, GoogleImageCrawler, BaiduImageCrawler


def crawl_into(dir_path: str, keyword: str, limit: int, engine: str):
    storage = {"root_dir": dir_path}
    if engine == "google":
        crawler = GoogleImageCrawler(storage=storage)
    elif engine == "baidu":
        crawler = BaiduImageCrawler(storage=storage)
    else:
        crawler = BingImageCrawler(storage=storage)
    crawler.crawl(keyword=keyword, max_num=limit)


def parse_args():
    ap = argparse.ArgumentParser("Download images only for class-2 (product + watermark)")
    ap.add_argument("--out-dir", type=str, default="data")
    ap.add_argument("--count", type=int, default=500, help="number of images to add to class 2")
    ap.add_argument("--per-keyword-limit", type=int, default=200)
    ap.add_argument("--engines", nargs="*", default=["bing", "google"], choices=["bing", "google", "baidu"])
    ap.add_argument("--keywords", nargs="*", default=[
        "watermarked product image",
        "brand watermark product",
        "stock photo watermark product",
        "watermark product photo",
        "商品 图片 水印",
        "水印 产品 照片",
        "品牌 水印 产品 图片",
    ])
    return ap.parse_args()


def main():
    args = parse_args()
    save_dir = os.path.join(args.out_dir, "train", "2")
    os.makedirs(save_dir, exist_ok=True)

    remaining = args.count
    while remaining > 0:
        for engine in args.engines:
            for kw in args.keywords:
                if remaining <= 0:
                    break
                limit = min(args.per_keyword_limit, remaining)
                crawl_into(save_dir, kw, limit, engine)
                remaining -= limit
            if remaining <= 0:
                break

    print(f"Downloaded {args.count} images (requested) into {save_dir}")


if __name__ == "__main__":
    main()