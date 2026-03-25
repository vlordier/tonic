"""Tests for the SEVAR dataset class."""

import os
import shutil
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

import tonic.datasets as datasets


def _make_fake_topics(n_events=50, n_imu=10, n_frames=3):
    """Return a fake importRosbag topics dict that resembles the SEVAR bag layout."""
    rng = np.random.default_rng(0)

    # Event streams: timestamps in seconds (monotonically increasing)
    ts = np.sort(rng.random(n_events).astype(float))
    x = rng.integers(0, 346, size=n_events)
    y = rng.integers(0, 260, size=n_events)
    pol = rng.integers(0, 2, size=n_events)

    def _events():
        return {
            "ts": ts.copy(),
            "x": x.copy(),
            "y": y.copy(),
            "pol": pol.copy(),
        }

    # Camera frames: list of (H, W, C) uint8 arrays
    frames = [rng.integers(0, 256, (480, 640, 3), dtype=np.uint8) for _ in range(n_frames)]

    # IMU timestamps
    imu_ts = np.sort(rng.random(n_imu).astype(float))

    return {
        "/davis/left/events": _events(),
        "/davis/right/events": _events(),
        "/camera/left": {"frames": list(frames)},
        "/camera/right": {"frames": list(frames)},
        "/imu/data": {"ts": imu_ts},
    }


class SEVARTestCase(unittest.TestCase):
    """Unit tests for tonic.datasets.SEVAR."""

    RECORDING = "AR-normal"

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        # Create the fake bag file so _check_exists can find it
        rec_dir = os.path.join(self.tmpdir, "SEVAR", self.RECORDING)
        os.makedirs(rec_dir, exist_ok=True)
        open(os.path.join(rec_dir, f"{self.RECORDING}.bag"), "w").close()

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _make_dataset(self, recording=None, **kwargs):
        if recording is None:
            recording = self.RECORDING
        return datasets.SEVAR(save_to=self.tmpdir, recording=recording, **kwargs)

    # ------------------------------------------------------------------
    # Construction tests
    # ------------------------------------------------------------------

    def test_single_recording_len(self):
        ds = self._make_dataset()
        self.assertEqual(len(ds), 1)

    def test_list_recording_len(self):
        # Create fake bag files for two recordings
        for rec in ["AR-normal", "AR-hdr"]:
            rec_dir = os.path.join(self.tmpdir, "SEVAR", rec)
            os.makedirs(rec_dir, exist_ok=True)
            open(os.path.join(rec_dir, f"{rec}.bag"), "w").close()

        ds = self._make_dataset(recording=["AR-normal", "AR-hdr"])
        self.assertEqual(len(ds), 2)

    def test_all_recordings_len(self):
        # Create fake bag files for all recordings
        for rec in datasets.SEVAR.recordings:
            rec_dir = os.path.join(self.tmpdir, "SEVAR", rec)
            os.makedirs(rec_dir, exist_ok=True)
            open(os.path.join(rec_dir, f"{rec}.bag"), "w").close()

        ds = self._make_dataset(recording="all")
        self.assertEqual(len(ds), len(datasets.SEVAR.recordings))

    def test_invalid_recording_raises(self):
        with self.assertRaises(RuntimeError):
            datasets.SEVAR(save_to=self.tmpdir, recording="nonexistent-recording")

    def test_missing_files_raises(self):
        shutil.rmtree(self.tmpdir)
        os.makedirs(self.tmpdir)
        with self.assertRaises(FileNotFoundError):
            datasets.SEVAR(save_to=self.tmpdir, recording=self.RECORDING)

    # ------------------------------------------------------------------
    # __getitem__ tests (importRosbag is mocked)
    # ------------------------------------------------------------------

    def _get_item(self, index=0, **kwargs):
        fake_topics = _make_fake_topics()
        ds = self._make_dataset(**kwargs)
        with patch("tonic.datasets.sevar.importRosbag", return_value=fake_topics):
            return ds[index]

    def test_getitem_returns_two_element_tuple(self):
        result = self._get_item()
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 2)

    def test_getitem_data_is_dict_with_expected_keys(self):
        data, _ = self._get_item()
        self.assertIsInstance(data, dict)
        for key in ("events_left", "events_right", "images_left", "images_right", "imu"):
            self.assertIn(key, data, msg=f"Key '{key}' missing from data dict")

    def test_getitem_events_have_correct_dtype(self):
        data, _ = self._get_item()
        self.assertEqual(data["events_left"].dtype, datasets.SEVAR.dtype)
        self.assertEqual(data["events_right"].dtype, datasets.SEVAR.dtype)

    def test_getitem_events_field_names(self):
        data, _ = self._get_item()
        for field in ("x", "y", "t", "p"):
            self.assertIn(field, data["events_left"].dtype.names)

    def test_getitem_images_are_ndarray(self):
        data, _ = self._get_item()
        self.assertIsInstance(data["images_left"], np.ndarray)
        self.assertIsInstance(data["images_right"], np.ndarray)

    def test_getitem_imu_is_dict_with_ts(self):
        data, _ = self._get_item()
        self.assertIn("ts", data["imu"])

    def test_getitem_target_none_without_gt_file(self):
        _, target = self._get_item()
        self.assertIsNone(target)

    def test_getitem_target_ndarray_with_gt_file(self):
        # Write a minimal TUM-format ground truth file
        rec_dir = os.path.join(self.tmpdir, "SEVAR", self.RECORDING)
        gt_path = os.path.join(rec_dir, f"{self.RECORDING}_gt.txt")
        with open(gt_path, "w") as f:
            f.write("# timestamp tx ty tz qx qy qz qw\n")
            f.write("0.0 0.1 0.2 0.3 0.0 0.0 0.0 1.0\n")
            f.write("0.1 0.4 0.5 0.6 0.0 0.0 0.0 1.0\n")

        _, target = self._get_item()
        self.assertIsInstance(target, np.ndarray)
        self.assertEqual(target.shape, (2, 8))

    def test_getitem_gt_ignores_comment_lines(self):
        rec_dir = os.path.join(self.tmpdir, "SEVAR", self.RECORDING)
        gt_path = os.path.join(rec_dir, f"{self.RECORDING}_gt.txt")
        with open(gt_path, "w") as f:
            f.write("# this is a comment\n")
            f.write("\n")
            f.write("1.0 0.0 0.0 0.0 0.0 0.0 0.0 1.0\n")

        _, target = self._get_item()
        self.assertEqual(target.shape, (1, 8))

    def test_getitem_events_timestamps_start_at_zero(self):
        """After normalisation events['t'] should start at 0."""
        data, _ = self._get_item()
        self.assertEqual(data["events_left"]["t"][0], 0)
        self.assertEqual(data["events_right"]["t"][0], 0)

    # ------------------------------------------------------------------
    # Transform tests
    # ------------------------------------------------------------------

    def test_transform_applied_to_data(self):
        sentinel = {"called": False}

        def my_transform(data):
            sentinel["called"] = True
            return data

        _ = self._get_item(transform=my_transform)
        self.assertTrue(sentinel["called"])

    def test_target_transform_applied_to_target(self):
        sentinel = {"called": False}

        def my_target_transform(target):
            sentinel["called"] = True
            return target

        _ = self._get_item(target_transform=my_target_transform)
        self.assertTrue(sentinel["called"])

    def test_transforms_applied_to_both(self):
        sentinel = {"called": False}

        def my_transforms(data, target):
            sentinel["called"] = True
            return data, target

        _ = self._get_item(transforms=my_transforms)
        self.assertTrue(sentinel["called"])


if __name__ == "__main__":
    unittest.main()
