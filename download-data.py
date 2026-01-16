#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import gzip
import shutil
import requests

BASE_DIR = "data/FashionMNIST/raw"

FILES = {
    "t10k-images-idx3-ubyte.gz":
        "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-images-idx3-ubyte.gz",
    "t10k-labels-idx1-ubyte.gz":
        "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-labels-idx1-ubyte.gz",
}


def download(url: str, dst_path: str):
    """Download file with streaming."""
    print(f"Downloading {url}")
    with requests.get(url, stream=True, timeout=30) as r:
        r.raise_for_status()
        with open(dst_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)


def ungzip(src: str, dst: str):
    """Unzip .gz file."""
    print(f"Extracting {src} -> {dst}")
    with gzip.open(src, "rb") as f_in:
        with open(dst, "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)


def main():
    os.makedirs(BASE_DIR, exist_ok=True)

    for name, url in FILES.items():
        gz_path = os.path.join(BASE_DIR, name)
        raw_path = os.path.join(BASE_DIR, name[:-3])

        if not os.path.exists(gz_path):
            download(url, gz_path)
        else:
            print(f"Skip download (exists): {gz_path}")

        if not os.path.exists(raw_path):
            ungzip(gz_path, raw_path)
        else:
            print(f"Skip extract (exists): {raw_path}")


if __name__ == "__main__":
    main()