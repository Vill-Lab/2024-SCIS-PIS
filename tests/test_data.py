"""Synthetic data checks; no research data or pretrained weights are needed."""

import json
import pickle
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from batches import IndexFlow, get_batches, save_dataset


JOINT_ORDER = [
    "rankle", "rknee", "rhip", "lhip", "lknee", "lankle", "rwrist", "relbow",
    "rshoulder", "lshoulder", "lelbow", "lwrist", "neck", "cnose", "leye",
    "reye", "lear", "rear",
]
POSE = np.array([
    [.35, .95], [.35, .75], [.4, .55], [.6, .55], [.65, .75], [.65, .95],
    [.2, .55], [.25, .4], [.35, .25], [.65, .25], [.75, .4], [.8, .55],
    [.5, .25], [.5, .15], [.54, .13], [.46, .13], [.58, .15], [.42, .15],
], dtype=np.float32)


class DataTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def make_index(self, identities=("0001", "0002", "0003"), counts=None,
                   reverse_order=False, windows_paths=False):
        counts = counts or [1] * len(identities)
        (self.root / "train").mkdir(exist_ok=True)
        names = []
        for number, (pid, count) in enumerate(zip(identities, counts)):
            for image_index in range(count):
                name = f"train/{pid}_c1_{image_index:04d}.png"
                Image.fromarray(np.full((128, 128, 3), 40 + number * 50,
                                        dtype=np.uint8)).save(self.root / name)
                names.append(name.replace("/", "\\") if windows_paths else name)
        order = JOINT_ORDER[::-1] if reverse_order else JOINT_ORDER
        pose = POSE[::-1] if reverse_order else POSE
        index = {"imgs": names, "train": [True] * len(names),
                 "joints": np.stack([pose] * len(names)), "joint_order": order}
        path = self.root / "index.p"
        with path.open("wb") as handle:
            pickle.dump(index, handle)
        return path

    def make_matches(self, pairs=None):
        path = self.root / "matches.json"
        path.write_text(json.dumps(pairs or {"0001": "0002", "0002": "0003",
                                            "0003": "0001"}), encoding="utf-8")
        return path

    def flow(self, path, **kwargs):
        return IndexFlow([5, 128, 128, 3], path, True, 1, shuffle=False, **kwargs)

    def test_training_needs_no_match_file_and_repeats_small_dataset(self):
        flow = self.flow(self.make_index(identities=("0001",)))
        batch = next(flow)
        self.assertEqual(len(batch), 7)
        self.assertEqual(batch[0].shape, (5, 128, 128, 3))
        self.assertEqual(batch[2].shape, (5, 64, 64, 24))
        np.testing.assert_array_equal(batch[4], [0, 0, 0, 0, 0])
        self.assertTrue(np.isfinite(batch[2]).all())
        self.assertEqual(next(flow)[0].shape[0], 5)

    def test_match_preserves_donor_then_source_order_and_final_partial_group(self):
        flow = self.flow(self.make_index(counts=[2, 1, 1]), sample="match",
                         match_path=self.make_matches(), batch_id_size=2)
        self.assertEqual(flow.n_groups, 2)
        first = next(flow)
        self.assertEqual(len(first), 6)
        self.assertEqual(first[5], [2, 1])
        self.assertEqual(list(first[4]), ["train/0002_c1_0000.png", "train/0003_c1_0000.png",
                                         "train/0001_c1_0000.png", "train/0001_c1_0001.png",
                                         "train/0002_c1_0000.png"])
        last = next(flow)
        self.assertEqual(last[5], [1])
        self.assertEqual(list(last[4]), ["train/0001_c1_0000.png", "train/0003_c1_0000.png"])
        with self.assertRaises(StopIteration):
            next(flow)

    def test_match_rejects_missing_same_and_unavailable_donors(self):
        path = self.make_index()
        for pairs in [{"0001": "0002"},
                      {"0001": "0001", "0002": "0003", "0003": "0001"},
                      {"0001": "9999", "0002": "0003", "0003": "0001"}]:
            with self.subTest(pairs=pairs), self.assertRaisesRegex(ValueError, "donor|match"):
                self.flow(path, sample="match", match_path=self.make_matches(pairs))

    def test_empty_split_has_actionable_error(self):
        with self.assertRaisesRegex(ValueError, "split|valid"):
            IndexFlow([1, 128, 128, 3], self.make_index(), False, 1)

    def test_joint_order_is_respected_and_windows_paths_load(self):
        normal = next(self.flow(self.make_index()))
        reordered = next(self.flow(self.make_index(reverse_order=True, windows_paths=True)))
        np.testing.assert_array_equal(normal[1], reordered[1])
        np.testing.assert_allclose(normal[2], reordered[2])

    def test_missing_joint_semantics_are_rejected(self):
        path = self.make_index()
        with path.open("rb") as handle:
            index = pickle.load(handle)
        index["joint_order"][-1] = "unknown"
        with path.open("wb") as handle:
            pickle.dump(index, handle)
        with self.assertRaisesRegex(ValueError, "joint"):
            self.flow(path)

    def test_buffered_match_exhaustion_stays_finite(self):
        flow = get_batches([5, 128, 128, 3], self.make_index(), True, 1,
                           sample="match", match_path=self.make_matches(), batch_id_size=2)
        try:
            self.assertEqual(flow.pids, ["0001", "0002", "0003"])
            self.assertEqual(len(list(flow)), 2)
            with self.assertRaises(StopIteration):
                next(flow)
        finally:
            flow.close()

    def test_save_uses_requested_directory_center_crop_and_complete_identity_prefix(self):
        images = np.zeros((2, 128, 128, 3), dtype=np.float32)
        images[:, :, :32] = -1
        images[:, :, 96:] = 1
        output = self.root / "images"
        save_dataset(images, output, ["train/0101_c1.png", "train\\12345_c2.png"])
        self.assertEqual(sorted(path.name for path in output.iterdir()),
                         ["13845_c2.png", "1601_c1.png"])
        with Image.open(output / "1601_c1.png") as image:
            self.assertEqual(image.size, (64, 128))
            self.assertTrue((np.array(image) == 127).all())

    def test_save_validates_counts_before_writing_any_image(self):
        output = self.root / "images"
        with self.assertRaisesRegex(ValueError, "count|number"):
            save_dataset(np.zeros((2, 128, 128, 3)), output, ["0001_c1.png"])
        self.assertFalse(output.exists())

    def test_reconstruction_zero_offset_preserves_leading_zero_identity(self):
        output = self.root / "reconstructions"
        save_dataset(np.zeros((1, 128, 128, 3)), output, ["train/0001_c1.png"],
                     identity_offset=0)
        self.assertEqual([path.name for path in output.iterdir()], ["0001_c1.png"])

    def test_existing_output_rejects_entire_later_batch_before_writing(self):
        output = self.root / "images"
        save_dataset(np.zeros((1, 128, 128, 3)), output, ["train/0001_c1.png"])
        original = (output / "1501_c1.png").read_bytes()
        with self.assertRaisesRegex(FileExistsError, "exist|overwrite"):
            save_dataset(np.ones((2, 128, 128, 3)), output,
                         ["train/0002_c1.png", "train/0001_c1.png"])
        self.assertEqual([path.name for path in output.iterdir()], ["1501_c1.png"])
        self.assertEqual((output / "1501_c1.png").read_bytes(), original)


if __name__ == "__main__":
    unittest.main()
