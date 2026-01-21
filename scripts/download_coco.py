#!/usr/bin/env python3
"""
COCO 2017 Dataset Download Script

This script downloads the COCO 2017 dataset required for training.

Usage:
    python scripts/download_coco.py
    python scripts/download_coco.py --data-dir /path/to/data
    python scripts/download_coco.py --val-only  # Download only validation set
"""

import argparse
import os
import sys
import zipfile
import hashlib
from pathlib import Path
from urllib.request import urlretrieve
from tqdm import tqdm


# COCO 2017 URLs and checksums
COCO_URLS = {
    'train2017': {
        'url': 'http://images.cocodataset.org/zips/train2017.zip',
        'size_gb': 18.0,
        'md5': None  # Too large for checksum
    },
    'val2017': {
        'url': 'http://images.cocodataset.org/zips/val2017.zip',
        'size_gb': 0.8,
        'md5': None
    },
    'annotations': {
        'url': 'http://images.cocodataset.org/annotations/annotations_trainval2017.zip',
        'size_gb': 0.25,
        'md5': None
    }
}


class DownloadProgressBar(tqdm):
    """Progress bar for downloads."""

    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_file(url: str, dest_path: str, desc: str = None):
    """Download file with progress bar."""
    with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc=desc) as t:
        urlretrieve(url, dest_path, reporthook=t.update_to)


def extract_zip(zip_path: str, extract_dir: str):
    """Extract zip file with progress bar."""
    print(f"Extracting {zip_path}...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        members = zip_ref.namelist()
        for member in tqdm(members, desc="Extracting"):
            zip_ref.extract(member, extract_dir)


def check_disk_space(path: str, required_gb: float) -> bool:
    """Check if there's enough disk space."""
    import shutil
    total, used, free = shutil.disk_usage(path)
    free_gb = free / (1024 ** 3)

    if free_gb < required_gb:
        print(f"Warning: Only {free_gb:.1f} GB free, but {required_gb:.1f} GB required")
        return False
    return True


def verify_dataset(data_dir: str) -> bool:
    """Verify that the dataset was downloaded correctly."""
    required_files = [
        'train2017',
        'val2017',
        'annotations/instances_train2017.json',
        'annotations/instances_val2017.json'
    ]

    all_exist = True
    for f in required_files:
        path = os.path.join(data_dir, f)
        if os.path.exists(path):
            if os.path.isdir(path):
                count = len(os.listdir(path))
                print(f"  ✓ {f}: {count} files")
            else:
                size_mb = os.path.getsize(path) / (1024 * 1024)
                print(f"  ✓ {f}: {size_mb:.1f} MB")
        else:
            print(f"  ✗ {f}: NOT FOUND")
            all_exist = False

    return all_exist


def parse_args():
    parser = argparse.ArgumentParser(description='Download COCO 2017 dataset')
    parser.add_argument('--data-dir', type=str, default='data/coco',
                        help='Directory to download dataset to')
    parser.add_argument('--val-only', action='store_true',
                        help='Download only validation set (for quick testing)')
    parser.add_argument('--keep-zip', action='store_true',
                        help='Keep downloaded zip files')
    parser.add_argument('--skip-download', action='store_true',
                        help='Skip download, only extract existing zips')
    return parser.parse_args()


def main():
    args = parse_args()

    data_dir = args.data_dir
    os.makedirs(data_dir, exist_ok=True)

    print("=" * 60)
    print("COCO 2017 Dataset Downloader")
    print("=" * 60)
    print(f"\nTarget directory: {os.path.abspath(data_dir)}")

    # Determine what to download
    if args.val_only:
        datasets = ['val2017', 'annotations']
        total_size = sum(COCO_URLS[d]['size_gb'] for d in datasets)
        print(f"Mode: Validation only (~{total_size:.1f} GB)")
    else:
        datasets = ['train2017', 'val2017', 'annotations']
        total_size = sum(COCO_URLS[d]['size_gb'] for d in datasets)
        print(f"Mode: Full dataset (~{total_size:.1f} GB)")

    # Check disk space
    print(f"\nChecking disk space...")
    if not check_disk_space(data_dir, total_size * 2):  # Need extra for extraction
        response = input("Continue anyway? (y/n): ")
        if response.lower() != 'y':
            sys.exit(1)

    # Download and extract each dataset
    for dataset in datasets:
        info = COCO_URLS[dataset]
        zip_name = f"{dataset}.zip" if dataset != 'annotations' else 'annotations_trainval2017.zip'
        zip_path = os.path.join(data_dir, zip_name)

        # Check if already extracted
        if dataset == 'annotations':
            check_path = os.path.join(data_dir, 'annotations', 'instances_train2017.json')
        else:
            check_path = os.path.join(data_dir, dataset)

        if os.path.exists(check_path):
            print(f"\n{dataset} already exists, skipping...")
            continue

        # Download
        if not args.skip_download:
            if os.path.exists(zip_path):
                print(f"\n{zip_name} already downloaded")
            else:
                print(f"\nDownloading {dataset} ({info['size_gb']:.1f} GB)...")
                try:
                    download_file(info['url'], zip_path, desc=dataset)
                except Exception as e:
                    print(f"Error downloading {dataset}: {e}")
                    print("\nAlternative: Download manually from:")
                    print(f"  {info['url']}")
                    continue

        # Extract
        if os.path.exists(zip_path):
            extract_zip(zip_path, data_dir)

            # Remove zip if not keeping
            if not args.keep_zip:
                print(f"Removing {zip_name}...")
                os.remove(zip_path)

    # Verify dataset
    print("\n" + "=" * 60)
    print("Verifying dataset...")
    print("=" * 60)

    if verify_dataset(data_dir):
        print("\n✓ Dataset download complete!")
        print(f"\nDataset location: {os.path.abspath(data_dir)}")
        print("\nYou can now run training:")
        print(f"  python scripts/train.py --data-dir {data_dir}")
    else:
        print("\n✗ Some files are missing. Please check the download.")

    # Print dataset statistics
    print("\n" + "=" * 60)
    print("Dataset Statistics (COCO 2017)")
    print("=" * 60)
    print("  Train images: ~118,287")
    print("  Val images: ~5,000")
    print("  Categories: 80")
    print("  Annotations: Object detection + segmentation")


if __name__ == '__main__':
    main()
