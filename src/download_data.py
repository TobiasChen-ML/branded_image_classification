import os
import argparse
from typing import List, Dict

from icrawler.builtin import BingImageCrawler, GoogleImageCrawler, BaiduImageCrawler


DEFAULT_KEYWORDS: Dict[int, List[str]] = {
    0: [
        "random street photo",
        "architecture building photo",
        "nature landscape photo",
        "pattern texture abstract",
    ],
    1: [
        "company logo icon",
        "brand logo vector simple",
        "logo mark black white",
        "minimal logo symbol",
    ],
    2: [
        "image watermark example",
        "stock photo watermark",
        "logo watermark overlay",
        "transparent watermark text",
    ],
}


def crawl_class(out_dir: str, cls: int, keywords: List[str], per_class: int, per_keyword_limit: int, engine: str = "bing"):
    save_dir = os.path.join(out_dir, "train", str(cls))
    os.makedirs(save_dir, exist_ok=True)
    remaining = per_class
    for kw in keywords:
        if remaining <= 0:
            break
        limit = min(per_keyword_limit, remaining)
        if engine == "google":
            crawler = GoogleImageCrawler(storage={"root_dir": save_dir})
        elif engine == "baidu":
            crawler = BaiduImageCrawler(storage={"root_dir": save_dir})
        else:
            crawler = BingImageCrawler(storage={"root_dir": save_dir})
        crawler.crawl(keyword=kw, max_num=limit)
        remaining -= limit


def parse_args():
    ap = argparse.ArgumentParser("Download images from internet into ImageFolder structure")
    ap.add_argument("--out-dir", type=str, default="data")
    ap.add_argument("--per-class", type=int, default=300)
    ap.add_argument("--per-keyword-limit", type=int, default=150)
    ap.add_argument("--engine", type=str, default="bing", choices=["bing", "google", "baidu"], help="search engine for crawling")
    ap.add_argument("--cls0", nargs="*", help="keywords for class 0 (non brand)")
    ap.add_argument("--cls1", nargs="*", help="keywords for class 1 (logo)")
    ap.add_argument("--cls2", nargs="*", help="keywords for class 2 (watermark)")
    return ap.parse_args()


def main():
    args = parse_args()

    kw0 = args.cls0 if args.cls0 else DEFAULT_KEYWORDS[0]
    kw1 = args.cls1 if args.cls1 else DEFAULT_KEYWORDS[1]
    kw2 = args.cls2 if args.cls2 else DEFAULT_KEYWORDS[2]

    crawl_class(args.out_dir, 0, kw0, args.per_class, args.per_keyword_limit, engine=args.engine)
    crawl_class(args.out_dir, 1, kw1, args.per_class, args.per_keyword_limit, engine=args.engine)
    crawl_class(args.out_dir, 2, kw2, args.per_class, args.per_keyword_limit, engine=args.engine)

    print(f"Downloaded images into {args.out_dir}/train/{0,1,2}")


if __name__ == "__main__":
    main()