"""Validate user-supplied image/pose annotations and write a PIS dataset index.

The manifest is a JSON object with imgs, joints, train, and joint_order fields.
Image paths are relative to the OUTPUT index directory, not the manifest.
Joint coordinates are normalized (x, y); negative values mark missing joints.
No pose estimation or image copying is performed.
"""
import argparse
import json
import os
import pickle
from pathlib import Path, PureWindowsPath
import sys
import tempfile

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from batches import DEFAULT_JOINT_ORDER, _identity


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate JSON key: " + key)
        result[key] = value
    return result


def read_json(path):
    with Path(path).open(encoding="utf-8-sig") as handle:
        return json.load(handle, object_pairs_hook=_unique_object)


def validate_index(data, index_path):
    """Return a validated index without loading any image pixels or TensorFlow."""
    if not isinstance(data, dict):
        raise ValueError("Index must be an object with imgs, joints, train, joint_order")
    missing = {"imgs", "joints", "train", "joint_order"} - data.keys()
    if missing:
        raise ValueError("Index is missing: " + ", ".join(sorted(missing)))
    names = data["imgs"]
    if not isinstance(names, (list, tuple, np.ndarray)) or not len(names):
        raise ValueError("imgs must be a nonempty list of relative image paths")
    order = data["joint_order"]
    if (not isinstance(order, (list, tuple, np.ndarray)) or len(order) != 18
            or any(not isinstance(part, str) for part in order)
            or set(order) != set(DEFAULT_JOINT_ORDER)):
        raise ValueError("joint_order must contain each of the 18 required names exactly once")
    try:
        raw_joints = np.asarray(data["joints"])
        if raw_joints.dtype.kind not in "fiu":
            raise ValueError("joints must contain numeric coordinates")
        if not np.isfinite(raw_joints).all() or np.any(raw_joints > 1):
            raise ValueError("joints must contain finite normalized xy values <= 1")
        with np.errstate(over="raise", invalid="raise"):
            joints = raw_joints.astype(np.float32)
    except (TypeError, ValueError, FloatingPointError, OverflowError) as error:
        raise ValueError("joints must be a numeric array with shape [N, 18, 2]") from error
    if joints.shape != (len(names), 18, 2):
        raise ValueError("joints must have shape [number of images, 18, 2]")
    if not np.isfinite(joints).all() or np.any(joints > 1):
        raise ValueError("joints must be finite normalized xy coordinates <= 1; negative values mark missing joints")
    flags = np.asarray(data["train"])
    if flags.shape != (len(names),) or flags.dtype.kind not in "biu" or not np.isin(flags, [0, 1]).all():
        raise ValueError("train must contain one boolean or integer 0/1 flag per image")
    base = Path(index_path).resolve().parent
    normalized = []
    for name in names:
        if not isinstance(name, str) or not name.strip():
            raise ValueError("imgs entries must be nonempty relative path strings")
        name = name.replace("\\", "/")
        if Path(name).is_absolute() or PureWindowsPath(name).drive or PureWindowsPath(name).root:
            raise ValueError("Image paths must be relative to the output index directory: " + name)
        _identity(name)
        if not (base / name).is_file():
            raise ValueError("Image does not exist relative to index directory: " + str(base / name))
        normalized.append(name)
    return {"imgs": normalized, "joints": joints, "train": flags.astype(bool),
            "joint_order": list(order)}


def atomic_write(path, content):
    """Replace output only after successful validation and complete serialization."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=path.name + ".", delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(content)
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", type=Path, required=True, help="JSON with imgs, joints, train, joint_order")
    parser.add_argument("--output", type=Path, required=True, help="Output index.p; image paths are relative to this directory")
    args = parser.parse_args(argv)
    try:
        if args.output.resolve() == args.manifest.resolve():
            raise ValueError("Output must differ from the source manifest")
        data = validate_index(read_json(args.manifest), args.output)
        if args.output.resolve() in {(args.output.parent / name).resolve() for name in data["imgs"]}:
            raise ValueError("Output must not replace a source image")
        atomic_write(args.output, pickle.dumps(data, protocol=pickle.HIGHEST_PROTOCOL))
    except (OSError, ValueError, TypeError) as error:
        parser.error(str(error))
    print("Wrote {} images to {}".format(len(data["imgs"]), args.output))


if __name__ == "__main__":
    main()
