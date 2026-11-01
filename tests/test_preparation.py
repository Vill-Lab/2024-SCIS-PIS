"""Data-preparation integration tests use generated images and annotations only."""
import copy
import json
import pickle
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.io import savemat

ROOT = Path(__file__).resolve().parents[1]
ORDER = [
    "rankle", "rknee", "rhip", "lhip", "lknee", "lankle", "rwrist", "relbow",
    "rshoulder", "lshoulder", "lelbow", "lwrist", "neck", "cnose", "leye",
    "reye", "lear", "rear",
]


class PreparationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.images = self.root / "data"
        self.images.mkdir()
        self.names = ["0002_c1.png", "0010_c1.png", "0007_c1.png", "0002_c2.png"]
        for name in self.names:
            Image.new("RGB", (32, 64), "gray").save(self.images / name)
        self.manifest = {
            "imgs": ["data/" + name for name in self.names],
            "joints": [[[.5, .5] for _ in ORDER] for _ in self.names],
            "train": [True] * len(self.names), "joint_order": ORDER,
        }
        self.index = self.root / "index.p"
        self.attributes = {
            "0002": {"age": 1, "hat": False, "gender": 0},
            "0010": {"age": 4, "hat": True, "gender": 1},
            "0007": {"age": 3, "hat": 1, "gender": True},
        }

    def tearDown(self):
        self.temp.cleanup()

    def run_tool(self, name, *args):
        return subprocess.run(
            [sys.executable, str(ROOT / "tools" / name), *map(str, args)],
            cwd=self.root, text=True, capture_output=True, timeout=30,
        )

    def prepare(self, manifest=None):
        # Manifest is deliberately in another directory: paths are index-relative.
        folder = self.root / "annotations"
        folder.mkdir(exist_ok=True)
        source = folder / "manifest.json"
        source.write_text(json.dumps(self.manifest if manifest is None else manifest), encoding="utf-8")
        return self.run_tool("prepare_index.py", "--manifest", source, "--output", self.index)

    def write_index(self):
        with self.index.open("wb") as handle:
            pickle.dump(self.manifest, handle)

    def match(self, attributes=None, output=None):
        source = self.root / "attributes.json"
        source.write_text(json.dumps(self.attributes if attributes is None else attributes), encoding="utf-8")
        return self.run_tool("make_matches.py", "--index", self.index,
                             "--attributes", source, "--output", output or self.root / "matches.json")

    def test_prepare_preserves_order_split_and_missing_joints_with_index_relative_paths(self):
        self.manifest["imgs"][0] = "data\\0002_c1.png"
        self.manifest["train"][-1] = False
        self.manifest["joints"][0][0] = [-1, -1]
        self.manifest["joint_order"] = ORDER[::-1]
        result = self.prepare()
        self.assertEqual(result.returncode, 0, result.stderr)
        with self.index.open("rb") as handle:
            data = pickle.load(handle)
        self.assertEqual(data["imgs"][0], "data/0002_c1.png")
        self.assertEqual(data["joint_order"], ORDER[::-1])
        self.assertEqual(np.asarray(data["joints"]).shape, (4, 18, 2))
        self.assertEqual(list(data["train"]), [True, True, True, False])
        np.testing.assert_array_equal(data["joints"][0][0], [-1, -1])

    def test_prepare_rejects_malformed_schema_coordinates_and_flags_without_output(self):
        bad = []
        for key in ["imgs", "joints", "train", "joint_order"]:
            value = copy.deepcopy(self.manifest)
            del value[key]
            bad.append(value)
        value = copy.deepcopy(self.manifest); value["train"] = [True]; bad.append(value)
        value = copy.deepcopy(self.manifest); value["joints"][0].pop(); bad.append(value)
        for coordinate in [1.1, 1.00000001, float("nan"), float("inf")]:
            value = copy.deepcopy(self.manifest); value["joints"][0][0][0] = coordinate; bad.append(value)
        value = copy.deepcopy(self.manifest); value["train"][0] = "false"; bad.append(value)
        value = copy.deepcopy(self.manifest); value["joint_order"] = ORDER[:-1] + [ORDER[0]]; bad.append(value)
        for manifest in bad:
            with self.subTest(manifest=manifest):
                result = self.prepare(manifest)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("error", result.stderr.lower())
                self.assertFalse(self.index.exists())

    def test_prepare_rejects_absolute_missing_and_idless_paths(self):
        for name in [str(self.images / self.names[0]), "C:\\data\\0002_c1.png",
                     "/data/0002_c1.png", "data/missing.png", "data/noidentity.png"]:
            if name == "data/noidentity.png":
                Image.new("RGB", (2, 2)).save(self.root / name)
            value = copy.deepcopy(self.manifest); value["imgs"][0] = name
            with self.subTest(path=name):
                result = self.prepare(value)
                self.assertNotEqual(result.returncode, 0)
                self.assertFalse(self.index.exists())

    def test_matches_preserve_leading_zeros_and_pick_first_eligible_index_identity(self):
        self.write_index()
        result = self.match()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads((self.root / "matches.json").read_text()),
                         {"0002": "0010", "0010": "0002", "0007": "0002"})

    def test_matches_exclude_validation_and_missing_core_joint_identities(self):
        self.manifest["train"][1] = False
        self.manifest["joints"][0][ORDER.index("lhip")] = [-1, -1]
        self.manifest["joint_order"] = ORDER[::-1]
        self.manifest["joints"] = [pose[::-1] for pose in self.manifest["joints"]]
        self.write_index()
        attributes = {pid: value for pid, value in self.attributes.items() if pid != "0010"}
        result = self.match(attributes)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads((self.root / "matches.json").read_text()),
                         {"0007": "0002", "0002": "0007"})

    def test_matches_fail_for_missing_attributes_or_no_donor_without_replacing_output(self):
        self.write_index()
        bad = copy.deepcopy(self.attributes)
        bad["0002"]["hat"] = True
        missing = {"2": self.attributes["0002"], "0010": self.attributes["0010"], "0007": self.attributes["0007"]}
        for attributes in [bad, missing]:
            with self.subTest(attributes=attributes):
                output = self.root / "matches.json"
                output.write_text("keep-me", encoding="utf-8")
                result = self.match(attributes)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("0002", result.stderr)
                self.assertEqual(output.read_text(), "keep-me")

    def test_matches_validate_binary_attributes_and_age_categories(self):
        self.write_index()
        for key, value in [("hat", 2), ("gender", "female"), ("age", []), ("age", True)]:
            attributes = copy.deepcopy(self.attributes)
            attributes["0002"][key] = value
            result = self.match(attributes)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn(key, result.stderr)
            self.assertFalse((self.root / "matches.json").exists())

    def test_matches_reject_empty_valid_training_split(self):
        self.manifest["train"] = [False] * len(self.names)
        self.write_index()
        result = self.match()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("train", result.stderr.lower())

    def test_market_mat_maps_image_index_not_row_or_directory_order(self):
        self.write_index()
        mat = self.root / "attributes.mat"
        savemat(mat, {"market_attribute": {"train": {
            "image_index": np.array(["0007", "0002", "0010"], dtype=object), "age": np.array([3, 1, 4]),
            "hat": np.array([2, 1, 2]), "gender": np.array([2, 1, 2]),
        }}})
        result = self.run_tool("make_matches.py", "--index", self.index,
                               "--market-attributes", mat, "--output", self.root / "matches.json")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads((self.root / "matches.json").read_text()),
                         {"0002": "0010", "0010": "0002", "0007": "0002"})

    def test_market_mat_rejects_missing_duplicate_and_invalid_codes(self):
        self.write_index()
        base = {"image_index": np.array(["0007", "0002", "0010"], dtype=object), "age": [3, 1, 4], "hat": [2, 1, 2], "gender": [2, 1, 2]}
        bad = []
        for key, value in [("image_index", np.array(["0007", "0002", "0002"], dtype=object)), ("age", [3, 1, 8]),
                           ("hat", [2, 0, 2]), ("gender", [2, 1])]:
            item = copy.deepcopy(base); item[key] = value; bad.append(item)
        item = copy.deepcopy(base); del item["image_index"]; bad.append(item)
        for item in bad:
            mat = self.root / "attributes.mat"
            savemat(mat, {"market_attribute": {"train": item}})
            result = self.run_tool("make_matches.py", "--index", self.index,
                                   "--market-attributes", mat, "--output", self.root / "matches.json")
            self.assertNotEqual(result.returncode, 0)
            self.assertFalse((self.root / "matches.json").exists())


if __name__ == "__main__":
    unittest.main()
