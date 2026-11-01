"""Synthetic pixel and pose checks for the original resize-and-pad preparation."""
import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import cv2
import numpy as np

from batches import DEFAULT_JOINT_ORDER

ROOT = Path(__file__).resolve().parents[1]


class ImagePreparationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.raw = self.root / "raw"
        (self.raw / "train").mkdir(parents=True)
        self.source = np.zeros((128, 64, 3), dtype=np.uint8)
        self.source[:, :, 0] = np.arange(64)
        self.source[:, :, 1] = np.arange(128)[:, None]
        self.source[:, :, 2] = 190
        cv2.imwrite(str(self.raw / "train/0001_c1.png"), self.source)
        pose = [[.5, .5] for _ in DEFAULT_JOINT_ORDER]
        pose[:5] = [[0, .25], [1, .75], [-1, -1], [-1, .5], [.5, -1]]
        self.data = {"imgs": ["train/0001_c1.png"], "joints": [pose],
                     "train": [True], "joint_order": DEFAULT_JOINT_ORDER}
        self.manifest = self.raw / "manifest.json"
        self.output = self.root / "prepared"

    def tearDown(self):
        self.temp.cleanup()

    def run_tool(self, output=None, data=None):
        self.manifest.write_text(json.dumps(self.data if data is None else data), encoding="utf-8")
        return subprocess.run(
            [sys.executable, str(ROOT / "tools/prepare_images.py"),
             "--manifest", str(self.manifest), "--output-dir", str(output or self.output)],
            cwd=self.root, capture_output=True, text=True, timeout=30,
        )

    def test_padding_replicates_edges_and_transforms_only_valid_pose_points(self):
        before = (self.raw / "train/0001_c1.png").read_bytes()
        result = self.run_tool()
        self.assertEqual(result.returncode, 0, result.stderr)
        output = cv2.imread(str(self.output / "train/0001_c1.png"))
        self.assertEqual(output.shape, (128, 128, 3))
        np.testing.assert_array_equal(output[:, 32:96], self.source)
        np.testing.assert_array_equal(output[:, :32], np.repeat(self.source[:, :1], 32, axis=1))
        np.testing.assert_array_equal(output[:, 96:], np.repeat(self.source[:, -1:], 32, axis=1))
        manifest = json.loads((self.output / "manifest.json").read_text())
        self.assertEqual(manifest["imgs"], ["train/0001_c1.png"])
        self.assertEqual(manifest["train"], [True])
        self.assertEqual(manifest["joint_order"], DEFAULT_JOINT_ORDER)
        np.testing.assert_array_equal(manifest["joints"][0][:5],
                                      [[.25, .25], [.75, .75], [-1, -1], [-1, .5], [.5, -1]])
        self.assertEqual((self.raw / "train/0001_c1.png").read_bytes(), before)
        # The output manifest is consumed directly by the existing index packer.
        packed = subprocess.run([sys.executable, str(ROOT / "tools/prepare_index.py"),
                                 "--manifest", str(self.output / "manifest.json"),
                                 "--output", str(self.output / "index.p")],
                                cwd=self.root, capture_output=True, text=True, timeout=30)
        self.assertEqual(packed.returncode, 0, packed.stderr)

    def test_resizes_nonstandard_images_to_required_dimensions(self):
        cv2.imwrite(str(self.raw / "train/0001_c1.png"), np.full((17, 9, 3), 73, dtype=np.uint8))
        result = self.run_tool()
        self.assertEqual(result.returncode, 0, result.stderr)
        np.testing.assert_array_equal(cv2.imread(str(self.output / "train/0001_c1.png")),
                                      np.full((128, 128, 3), 73, dtype=np.uint8))

    def test_rejects_existing_output_and_source_directory_without_overwriting(self):
        self.output.mkdir()
        marker = self.output / "keep.txt"
        marker.write_text("unchanged", encoding="utf-8")
        for destination in [self.output, self.raw]:
            result = self.run_tool(output=destination)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("exist", result.stderr.lower())
        self.assertEqual(marker.read_text(), "unchanged")
        self.assertFalse((self.output / "manifest.json").exists())
        np.testing.assert_array_equal(cv2.imread(str(self.raw / "train/0001_c1.png")), self.source)

    def test_rejects_paths_outside_manifest_root_and_duplicate_targets(self):
        cv2.imwrite(str(self.root / "0002_c1.png"), self.source)
        for name in ["../0002_c1.png", "train/../../0002_c1.png", str(self.root / "0002_c1.png")]:
            with self.subTest(name=name):
                data = copy.deepcopy(self.data); data["imgs"] = [name]
                result = self.run_tool(data=data)
                self.assertNotEqual(result.returncode, 0)
                self.assertFalse(self.output.exists())
        data = copy.deepcopy(self.data)
        for key in ["imgs", "joints", "train"]:
            data[key] = data[key] * 2
        result = self.run_tool(data=data)
        self.assertNotEqual(result.returncode, 0)
        self.assertFalse(self.output.exists())

    def test_bad_image_does_not_publish_partial_output(self):
        (self.raw / "train/0002_c1.png").write_bytes(b"invalid-image")
        data = copy.deepcopy(self.data)
        data["imgs"].append("train/0002_c1.png")
        data["joints"].append(data["joints"][0])
        data["train"].append(False)
        result = self.run_tool(data=data)
        self.assertNotEqual(result.returncode, 0)
        self.assertFalse(self.output.exists())


if __name__ == "__main__":
    unittest.main()
