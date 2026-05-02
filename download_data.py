"""
download_data.py
----------------
Helper script to automatically download the Goodbooks-10k dataset.

The Goodbooks-10k dataset contains:
  - 10,000 books with metadata (title, author, ISBN, etc.)
  - ~6 million ratings from ~53,000 users
  - Rating scale: 1 to 5 stars

Source: https://github.com/zygmuntz/goodbooks-10k
The raw CSV files are hosted on GitHub and freely available.

Run this file with:
  python download_data.py
"""

import os
import requests


# ─── File URLs (from GitHub raw content) ──────────────────────────────────────
DATA_FILES = {
    "books.csv": (
        "https://raw.githubusercontent.com/"
        "zygmuntz/goodbooks-10k/master/books.csv"
    ),
    "ratings.csv": (
        "https://raw.githubusercontent.com/"
        "zygmuntz/goodbooks-10k/master/ratings.csv"
    ),
}

DATA_DIR = "data"


def download_file(filename: str, url: str):
    """
    Download a file from a URL and save it to the data/ directory.

    Shows a progress indicator during download.
    Skips the download if the file already exists.
    """
    filepath = os.path.join(DATA_DIR, filename)

    # Skip if already downloaded
    if os.path.exists(filepath):
        size_mb = os.path.getsize(filepath) / (1024 * 1024)
        print(f"  [SKIP] {filename} already exists ({size_mb:.1f} MB)")
        return

    print(f"  [DOWNLOADING] {filename}...")
    print(f"  URL: {url}")

    try:
        # stream=True → download in chunks (memory efficient for large files)
        response = requests.get(url, stream=True, timeout=60)
        response.raise_for_status()  # Raise an error if download failed

        total_size = int(response.headers.get("content-length", 0))
        downloaded = 0
        chunk_size = 8192  # 8 KB per chunk

        with open(filepath, "wb") as f:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)

                    # Show simple progress: downloaded / total MB
                    if total_size > 0:
                        pct = (downloaded / total_size) * 100
                        mb = downloaded / (1024 * 1024)
                        print(f"  → {mb:.1f} MB / {total_size/1024/1024:.1f} MB ({pct:.0f}%)", end="\r")

        print(f"\n  [DONE] Saved to '{filepath}'")

    except requests.exceptions.ConnectionError:
        print(f"  [ERROR] Could not connect. Check your internet connection.")
        raise
    except requests.exceptions.Timeout:
        print(f"  [ERROR] Download timed out. Try again.")
        raise
    except Exception as e:
        print(f"  [ERROR] Failed to download {filename}: {e}")
        raise


if __name__ == "__main__":
    print("=" * 55)
    print("  Goodbooks-10k Dataset Downloader")
    print("=" * 55)

    # Create data/ directory if it doesn't exist
    os.makedirs(DATA_DIR, exist_ok=True)
    print(f"Saving files to: ./{DATA_DIR}/\n")

    # Download each file
    for filename, url in DATA_FILES.items():
        download_file(filename, url)
        print()

    print("=" * 55)
    print("  All files downloaded! Next step:")
    print("  $ python train.py")
    print("=" * 55)
