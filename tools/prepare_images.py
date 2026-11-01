"""Prepare raw ReID images and poses using the original PIS resize2128 step.

Source image paths are relative to the input manifest directory. Images are
resized to width 64, height 128, then padded by 32 replicated pixels on each side.
For valid pose points, x becomes 0.5*x + 0.25; y and missing points are unchanged.
The output directory must not exist. No pose estimation is performed.
"""
import argparse
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import sys
import tempfile

import cv2
import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.prepare_index import read_json, validate_index


def prepare_images(manifest_path, output_dir):
    manifest_path = Path(manifest_path).resolve()
    output = Path(output_dir).absolute()
    if os.path.lexists(output):
        raise ValueError("Output directory already exists; choose a new directory: " + str(output))
    source_root = manifest_path.parent
    raw = read_json(manifest_path)
    index = validate_index(raw, manifest_path)
    destinations = set()
    for name in index["imgs"]:
        relative = PurePosixPath(name)
        if ".." in relative.parts:
            raise ValueError("Image paths must not traverse outside the manifest directory: " + name)
        if relative.parts[0].lower() == "manifest.json":
            raise ValueError("Image path conflicts with the output manifest: " + name)
        source = (source_root / name).resolve()
        if not source.is_relative_to(source_root):
            raise ValueError("Image path resolves outside the manifest directory: " + name)
        destination = (output / name).resolve()
        if not destination.is_relative_to(output.resolve()):
            raise ValueError("Image path resolves outside the output directory: " + name)
        key = os.path.normcase(str(destination))
        if key in destinations:
            raise ValueError("Duplicate output image path: " + name)
        destinations.add(key)

    # Keep original annotation precision, including all coordinates of missing points.
    joints = np.asarray(raw["joints"], dtype=np.float64).copy()
    valid = (joints >= 0).all(axis=2)
    joints[:, :, 0][valid] = 0.5 * joints[:, :, 0][valid] + 0.25
    prepared = {"imgs": index["imgs"], "joints": joints.tolist(),
                "train": index["train"].tolist(), "joint_order": index["joint_order"]}

    output.parent.mkdir(parents=True, exist_ok=True)
    # Stage before creating the final directory so an unreadable image cannot
    # leave a partially prepared dataset. The staging tree is confined here.
    with tempfile.TemporaryDirectory(prefix=".pis-images-", dir=output.parent) as temporary:
        staging = Path(temporary).resolve()
        if not staging.is_relative_to(output.parent.resolve()):
            raise ValueError("Staging directory must stay inside the output parent")
        for name in index["imgs"]:
            image = cv2.imread(str(source_root / name), cv2.IMREAD_COLOR)
            if image is None:
                raise ValueError("Cannot read image: " + str(source_root / name))
            resized = cv2.resize(image, (64, 128), interpolation=cv2.INTER_LINEAR)
            padded = cv2.copyMakeBorder(resized, 0, 0, 32, 32, cv2.BORDER_REPLICATE)
            destination = staging / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            if not cv2.imwrite(str(destination), padded):
                raise ValueError("Cannot encode output image: " + name)
        (staging / "manifest.json").write_text(json.dumps(prepared, indent=2) + "\n", encoding="utf-8")

        # Exclusive directory creation and exclusive file writes prevent overwrites,
        # including a destination created while source images were being processed.
        output.mkdir(exist_ok=False)
        for name in [*index["imgs"], "manifest.json"]:
            destination = output / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            if not destination.resolve().is_relative_to(output.resolve()):
                raise ValueError("Output path resolves outside the output directory: " + name)
            with (staging / name).open("rb") as source, destination.open("xb") as target:
                shutil.copyfileobj(source, target)
    return len(index["imgs"])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", type=Path, required=True,
                        help="Raw pose manifest; imgs are relative to its directory")
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="New output directory for padded images and manifest.json")
    args = parser.parse_args(argv)
    try:
        count = prepare_images(args.manifest, args.output_dir)
    except (OSError, ValueError, TypeError, cv2.error) as error:
        parser.error(str(error))
    print("Prepared {} images and {}".format(count, args.output_dir / "manifest.json"))


if __name__ == "__main__":
    main()
