"""
Convert every JPEG in images_02/BOOT-*/ to PNG with the same base name.

    images_02/BOOT-005/01_side_right.jpeg  ->  images_02/BOOT-005/01_side_right.png

Usage:
    python convert_jpeg_to_png.py                    # uses ./images_02
    python convert_jpeg_to_png.py path/to/images_02
    python convert_jpeg_to_png.py images_02 --delete-original
    python convert_jpeg_to_png.py images_02 --overwrite

Requires: pip install pillow
"""

import argparse
from pathlib import Path

from PIL import Image, ImageOps

JPEG_EXTENSIONS = {".jpg", ".jpeg"}


def convert_one(src: Path, overwrite: bool, delete_original: bool) -> str:
    dst = src.with_suffix(".png")

    if dst.exists() and not overwrite:
        return f"SKIP     {src}  (already exists: {dst.name})"

    with Image.open(src) as img:
        # Apply EXIF rotation; PNG doesn't keep the orientation tag,
        # so without this some phone photos would come out sideways.
        img = ImageOps.exif_transpose(img)

        # CMYK / other JPEG modes aren't valid for PNG
        if img.mode not in ("RGB", "RGBA", "L", "LA"):
            img = img.convert("RGB")

        img.save(dst, format="PNG", optimize=True)

    if delete_original:
        src.unlink()

    return f"OK       {src}  ->  {dst.name}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert JPEG images to PNG.")
    parser.add_argument("root", nargs="?", default="images_02",
                        help="Root directory containing BOOT-* folders (default: images_02)")
    parser.add_argument("--overwrite", action="store_true",
                        help="Overwrite a PNG that already exists with the same name")
    parser.add_argument("--delete-original", action="store_true",
                        help="Delete the JPEG after successful conversion")
    args = parser.parse_args()

    root = Path(args.root)
    if not root.is_dir():
        raise SystemExit(f"Directory not found: {root.resolve()}")

    # rglob covers BOOT-001, BOOT-002, ... and any nested folders
    jpegs = sorted(p for p in root.rglob("*")
                   if p.is_file() and p.suffix.lower() in JPEG_EXTENSIONS)

    if not jpegs:
        print(f"No JPEG files found under {root.resolve()}")
        return

    converted = skipped = failed = 0
    for src in jpegs:
        try:
            msg = convert_one(src, args.overwrite, args.delete_original)
            print(msg)
            if msg.startswith("OK"):
                converted += 1
            else:
                skipped += 1
        except Exception as e:  # keep going if one file is corrupt
            print(f"FAIL     {src}  ({e})")
            failed += 1

    print(f"\nDone. Converted: {converted}, Skipped: {skipped}, Failed: {failed}")


if __name__ == "__main__":
    main()