"""Create one attribute-different donor per valid training identity.

Candidates follow their first appearance in the index. A donor must differ in
age category, hat, and gender. Every source requires a donor; no fallback is used.
Only load trusted pickle indices. No images or attributes are downloaded.
"""
import argparse
import json
from pathlib import Path
import pickle
import sys

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from batches import _identity
from tools.prepare_index import atomic_write, read_json, validate_index


def training_identities(index):
    """Match IndexFlow's train and visible-shoulder/hip filtering and ordering."""
    core = [index["joint_order"].index(name) for name in
            ("lshoulder", "rshoulder", "lhip", "rhip")]
    result = {}
    for name, pose, train in zip(index["imgs"], index["joints"], index["train"]):
        if train and (pose[core] >= 0).all():
            result.setdefault(_identity(name), None)
    if not result:
        raise ValueError("No valid training identities; check train flags and shoulder/hip joints")
    return list(result)


def validate_attributes(attributes, identities):
    if not isinstance(attributes, dict):
        raise ValueError("Attributes must map identity strings to age, hat, gender objects")
    result = {}
    age_type = None
    for pid in identities:
        value = attributes.get(pid)
        if not isinstance(value, dict):
            raise ValueError("Missing attributes for identity " + pid + "; preserve leading zeros")
        age = value.get("age")
        if type(age) not in (int, str) or (isinstance(age, str) and not age.strip()):
            raise ValueError("age must be an integer or nonempty category string for identity " + pid)
        if age_type is not None and type(age) is not age_type:
            raise ValueError("age categories must use the same type for all training identities")
        age_type = type(age)
        binary = []
        for key in ("hat", "gender"):
            attribute = value.get(key)
            if type(attribute) not in (bool, int) or attribute not in (0, 1):
                raise ValueError(key + " must be boolean or integer 0/1 for identity " + pid)
            binary.append(bool(attribute))
        result[pid] = (age, *binary)
    return result


def load_market_attributes(path):
    """Read official Market-1501_Attribute market_attribute.train annotations.

    image_index contains identity strings (including leading zeros). Official
    age codes are 1..4; hat/gender codes are 1..2, converted to 0..1 here.
    Source: https://github.com/vana77/Market-1501_Attribute
    """
    from scipy.io import loadmat
    document = loadmat(path, simplify_cells=True)
    try:
        train = document["market_attribute"]["train"]
        columns = {key: np.atleast_1d(train[key]) for key in
                   ("image_index", "age", "hat", "gender")}
    except (KeyError, TypeError, IndexError) as error:
        raise ValueError("MAT must contain market_attribute.train with image_index, age, hat, gender") from error
    count = len(columns["image_index"])
    if not count or any(value.shape != (count,) for value in columns.values()):
        raise ValueError("MAT image_index, age, hat, gender must be equally sized vectors")
    for key, allowed in [("age", [1, 2, 3, 4]), ("hat", [1, 2]), ("gender", [1, 2])]:
        values = columns[key]
        if values.dtype.kind not in "fiu" or not np.isin(values, allowed).all():
            raise ValueError("Invalid official Market MAT " + key + " codes")
    attributes = {}
    for row, pid in enumerate(columns["image_index"]):
        if not isinstance(pid, str) or not pid or not pid.isascii() or not pid.isdigit():
            raise ValueError("MAT image_index must contain original identity strings with leading zeros")
        if pid in attributes:
            raise ValueError("Duplicate MAT image_index identity: " + pid)
        attributes[pid] = {"age": int(columns["age"][row]),
                           "hat": int(columns["hat"][row]) - 1,
                           "gender": int(columns["gender"][row]) - 1}
    return attributes


def choose_matches(identities, attributes):
    values = validate_attributes(attributes, identities)
    matches = {}
    unmatched = []
    for source in identities:
        for donor in identities:
            if source != donor and all(left != right for left, right in zip(values[source], values[donor])):
                matches[source] = donor
                break
        else:
            unmatched.append(source)
    if unmatched:
        raise ValueError("No donor with different age, hat, and gender for identities: " + ", ".join(unmatched))
    return matches


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--index", type=Path, required=True, help="Trusted PIS index.p")
    sources = parser.add_mutually_exclusive_group(required=True)
    sources.add_argument("--attributes", type=Path, help="JSON: pid -> {age: category, hat: bool/0/1, gender: bool/0/1}")
    sources.add_argument("--market-attributes", type=Path, help="Official market_attribute.mat; uses the train split and image_index")
    parser.add_argument("--output", type=Path, required=True, help="Output matches.json")
    args = parser.parse_args(argv)
    try:
        annotation_path = args.attributes or args.market_attributes
        if args.output.resolve() in {args.index.resolve(), annotation_path.resolve()}:
            raise ValueError("Output must differ from index and attribute inputs")
        with args.index.open("rb") as handle:
            index = validate_index(pickle.load(handle), args.index)
        if args.output.resolve() in {(args.index.parent / name).resolve() for name in index["imgs"]}:
            raise ValueError("Output must not replace a source image")
        identities = training_identities(index)
        attributes = (read_json(args.attributes) if args.attributes else
                      load_market_attributes(args.market_attributes))
        matches = choose_matches(identities, attributes)
        atomic_write(args.output, (json.dumps(matches, indent=2) + "\n").encode("utf-8"))
    except (OSError, ValueError, TypeError, pickle.UnpicklingError, EOFError) as error:
        parser.error(str(error))
    print("Wrote {} identity pairs to {}".format(len(matches), args.output))


if __name__ == "__main__":
    main()
