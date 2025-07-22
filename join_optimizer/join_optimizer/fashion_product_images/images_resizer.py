"""
Resize all images in the `images` directory so that the longest side
is at most 1000 px, saving the result into `images_resized`.
Only images whose largest dimension exceeds 1000 px are actually
resized; smaller images are copied unchanged.

The script uses a thread pool to process several images in parallel,
which is usually faster than a purely sequential pass because I/O
and JPEG/PNG compression/decompression release the GIL in Pillow.
"""

import os
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

from PIL import Image

# -----------------------------------------------------------------------------#
# Configuration – edit if you like                                             #
# -----------------------------------------------------------------------------#

SRC_DIR = Path("images")
DST_DIR = Path("images_resized")
MAX_DIM = 1000  # pixels
THREADS = os.cpu_count() or 4  # default: 1 thread per logical CPU

# Allowed image file suffixes (case‑insensitive)
IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff"}


def resize_copy(src_path: Path) -> str:
    """
    Resize *src_path* into DST_DIR/src_path.name if necessary, keeping
    the original aspect ratio and image format. Returns a short status
    string for logging.
    """
    try:
        with Image.open(src_path) as im:
            width, height = im.size
            dst_path = DST_DIR / src_path.name

            if max(width, height) <= MAX_DIM:
                # No resize needed – just copy (avoid re‑encoding if possible)
                im.save(dst_path, format=im.format)
                return f"copied  {src_path.name} ({width}×{height})"
            else:
                # Pillow’s thumbnail keeps aspect ratio & is in‑place
                im.thumbnail((MAX_DIM, MAX_DIM), Image.LANCZOS)
                im.save(
                    dst_path,
                    format=im.format,
                    optimize=True,
                    quality=85 if im.format in {"JPEG", "JPG"} else None,
                )
                new_w, new_h = im.size
                return (
                    f"resized {src_path.name} "
                    f"{width}×{height} → {new_w}×{new_h}"
                )
    except Exception as exc:  # pragma: no cover
        return f"error   {src_path.name}: {exc}"


def main() -> None:
    if not SRC_DIR.exists():
        raise SystemExit(f"Source directory {SRC_DIR} does not exist.")
    DST_DIR.mkdir(parents=True, exist_ok=True)

    candidates = [
        p
        for p in SRC_DIR.iterdir()
        if p.is_file() and p.suffix.lower() in IMG_EXTS
    ]
    if not candidates:
        raise SystemExit("No images found to process.")

    print(
        f"Processing {len(candidates)} image(s) with "
        f"{THREADS} worker thread(s)…"
    )

    with ThreadPoolExecutor(max_workers=THREADS) as pool:
        futures = {pool.submit(resize_copy, p): p for p in candidates}
        for fut in as_completed(futures):
            print(fut.result())

    print("Done ✅")


if __name__ == "__main__":
    main()
