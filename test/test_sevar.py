"""Tests for the SEVAR dataset class."""

import os
import shutil
import tempfile
from unittest.mock import patch

import numpy as np
import pytest

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


RECORDING = "AR-normal"


@pytest.fixture
def tmpdir_with_bag():
    """Temp directory that contains a fake bag file for AR-normal."""
    tmpdir = tempfile.mkdtemp()
    rec_dir = os.path.join(tmpdir, "SEVAR", RECORDING)
    os.makedirs(rec_dir, exist_ok=True)
    open(os.path.join(rec_dir, f"{RECORDING}.bag"), "w").close()
    yield tmpdir
    shutil.rmtree(tmpdir, ignore_errors=True)


def _make_dataset(tmpdir, recording=None, **kwargs):
    if recording is None:
        recording = RECORDING
    return datasets.SEVAR(save_to=tmpdir, recording=recording, **kwargs)


def _get_item(tmpdir, index=0, **kwargs):
    fake_topics = _make_fake_topics()
    ds = _make_dataset(tmpdir, **kwargs)
    with patch("tonic.datasets.sevar.importRosbag", return_value=fake_topics):
        return ds[index]


# ------------------------------------------------------------------
# Construction tests
# ------------------------------------------------------------------


def test_single_recording_len(tmpdir_with_bag):
    ds = _make_dataset(tmpdir_with_bag)
    assert len(ds) == 1


def test_list_recording_len(tmpdir_with_bag):
    for rec in ["AR-normal", "AR-hdr"]:
        rec_dir = os.path.join(tmpdir_with_bag, "SEVAR", rec)
        os.makedirs(rec_dir, exist_ok=True)
        open(os.path.join(rec_dir, f"{rec}.bag"), "w").close()

    ds = _make_dataset(tmpdir_with_bag, recording=["AR-normal", "AR-hdr"])
    assert len(ds) == 2


def test_all_recordings_len(tmpdir_with_bag):
    for rec in datasets.SEVAR.recordings:
        rec_dir = os.path.join(tmpdir_with_bag, "SEVAR", rec)
        os.makedirs(rec_dir, exist_ok=True)
        open(os.path.join(rec_dir, f"{rec}.bag"), "w").close()

    ds = _make_dataset(tmpdir_with_bag, recording="all")
    assert len(ds) == len(datasets.SEVAR.recordings)


def test_invalid_recording_raises(tmpdir_with_bag):
    with pytest.raises(RuntimeError):
        datasets.SEVAR(save_to=tmpdir_with_bag, recording="nonexistent-recording")


def test_missing_files_raises():
    tmpdir = tempfile.mkdtemp()
    try:
        with pytest.raises(FileNotFoundError):
            datasets.SEVAR(save_to=tmpdir, recording=RECORDING)
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


# ------------------------------------------------------------------
# __getitem__ tests (importRosbag is mocked)
# ------------------------------------------------------------------


def test_getitem_returns_two_element_tuple(tmpdir_with_bag):
    result = _get_item(tmpdir_with_bag)
    assert isinstance(result, tuple)
    assert len(result) == 2


def test_getitem_data_is_dict_with_expected_keys(tmpdir_with_bag):
    data, _ = _get_item(tmpdir_with_bag)
    assert isinstance(data, dict)
    for key in ("events_left", "events_right", "images_left", "images_right", "imu"):
        assert key in data, f"Key '{key}' missing from data dict"


def test_getitem_events_have_correct_dtype(tmpdir_with_bag):
    data, _ = _get_item(tmpdir_with_bag)
    assert data["events_left"].dtype == datasets.SEVAR.dtype
    assert data["events_right"].dtype == datasets.SEVAR.dtype


def test_getitem_events_field_names(tmpdir_with_bag):
    data, _ = _get_item(tmpdir_with_bag)
    for field in ("x", "y", "t", "p"):
        assert field in data["events_left"].dtype.names


def test_getitem_images_are_ndarray(tmpdir_with_bag):
    data, _ = _get_item(tmpdir_with_bag)
    assert isinstance(data["images_left"], np.ndarray)
    assert isinstance(data["images_right"], np.ndarray)


def test_getitem_imu_is_dict_with_ts(tmpdir_with_bag):
    data, _ = _get_item(tmpdir_with_bag)
    assert "ts" in data["imu"]


def test_getitem_target_none_without_gt_file(tmpdir_with_bag):
    _, target = _get_item(tmpdir_with_bag)
    assert target is None


def test_getitem_target_ndarray_with_gt_file(tmpdir_with_bag):
    rec_dir = os.path.join(tmpdir_with_bag, "SEVAR", RECORDING)
    gt_path = os.path.join(rec_dir, f"{RECORDING}_gt.txt")
    with open(gt_path, "w") as f:
        f.write("# timestamp tx ty tz qx qy qz qw\n")
        f.write("0.0 0.1 0.2 0.3 0.0 0.0 0.0 1.0\n")
        f.write("0.1 0.4 0.5 0.6 0.0 0.0 0.0 1.0\n")

    _, target = _get_item(tmpdir_with_bag)
    assert isinstance(target, np.ndarray)
    assert target.shape == (2, 8)


def test_getitem_gt_ignores_comment_lines(tmpdir_with_bag):
    rec_dir = os.path.join(tmpdir_with_bag, "SEVAR", RECORDING)
    gt_path = os.path.join(rec_dir, f"{RECORDING}_gt.txt")
    with open(gt_path, "w") as f:
        f.write("# this is a comment\n")
        f.write("\n")
        f.write("1.0 0.0 0.0 0.0 0.0 0.0 0.0 1.0\n")

    _, target = _get_item(tmpdir_with_bag)
    assert target.shape == (1, 8)


def test_getitem_events_timestamps_start_at_zero(tmpdir_with_bag):
    """After normalisation events['t'] should start at 0."""
    data, _ = _get_item(tmpdir_with_bag)
    assert data["events_left"]["t"][0] == 0
    assert data["events_right"]["t"][0] == 0


# ------------------------------------------------------------------
# Transform tests
# ------------------------------------------------------------------


def test_transform_applied_to_data(tmpdir_with_bag):
    sentinel = {"called": False}

    def my_transform(data):
        sentinel["called"] = True
        return data

    _get_item(tmpdir_with_bag, transform=my_transform)
    assert sentinel["called"]


def test_target_transform_applied_to_target(tmpdir_with_bag):
    sentinel = {"called": False}

    def my_target_transform(target):
        sentinel["called"] = True
        return target

    _get_item(tmpdir_with_bag, target_transform=my_target_transform)
    assert sentinel["called"]


def test_transforms_applied_to_both(tmpdir_with_bag):
    sentinel = {"called": False}

    def my_transforms(data, target):
        sentinel["called"] = True
        return data, target

    _get_item(tmpdir_with_bag, transforms=my_transforms)
    assert sentinel["called"]


# ------------------------------------------------------------------
# data_selection tests
# ------------------------------------------------------------------


def test_data_selection_events_only(tmpdir_with_bag):
    """Requesting only event streams must omit images and IMU."""
    data, _ = _get_item(tmpdir_with_bag, data_selection=["events_left", "events_right"])
    assert "events_left" in data
    assert "events_right" in data
    assert "images_left" not in data
    assert "images_right" not in data
    assert "imu" not in data


def test_data_selection_single_string(tmpdir_with_bag):
    """A single string should be accepted in addition to a list."""
    data, _ = _get_item(tmpdir_with_bag, data_selection="events_left")
    assert "events_left" in data
    assert "events_right" not in data


def test_data_selection_images_only(tmpdir_with_bag):
    data, _ = _get_item(tmpdir_with_bag, data_selection=["images_left", "images_right"])
    assert "images_left" in data
    assert "images_right" in data
    assert "events_left" not in data


def test_data_selection_imu_only(tmpdir_with_bag):
    data, _ = _get_item(tmpdir_with_bag, data_selection="imu")
    assert "imu" in data
    assert "events_left" not in data


def test_data_selection_invalid_key_raises(tmpdir_with_bag):
    with pytest.raises(ValueError, match="Invalid data_selection key"):
        _make_dataset(tmpdir_with_bag, data_selection=["nonexistent_key"])


def test_data_selection_default_returns_all_keys(tmpdir_with_bag):
    """Default (None) must return all five data streams."""
    data, _ = _get_item(tmpdir_with_bag)
    for key in ("events_left", "events_right", "images_left", "images_right", "imu"):
        assert key in data


def test_data_selection_events_correct_dtype(tmpdir_with_bag):
    """Even when selected via data_selection, events must have the right dtype."""
    data, _ = _get_item(tmpdir_with_bag, data_selection=["events_left"])
    assert data["events_left"].dtype == datasets.SEVAR.dtype


# ------------------------------------------------------------------
# download_instructions tests
# ------------------------------------------------------------------


def test_download_instructions_runs_without_error():
    """download_instructions() must be callable as a classmethod."""
    datasets.SEVAR.download_instructions()


def test_download_instructions_with_selection():
    """download_instructions() must accept an explicit list of recordings."""
    datasets.SEVAR.download_instructions(selection=["AR-normal", "AR-hdr"])


def test_baidu_resources_complete():
    """Every recording must have a Baidu Pan entry."""
    for rec in datasets.SEVAR.recordings:
        assert rec in datasets.SEVAR.baidu_resources, (
            f"Missing Baidu Pan URL for recording '{rec}'"
        )


def test_onedrive_gt_resources_complete():
    """Every recording must have an OneDrive GT URL entry."""
    for rec in datasets.SEVAR.recordings:
        assert rec in datasets.SEVAR.onedrive_gt_resources, (
            f"Missing OneDrive GT URL for recording '{rec}'"
        )


def test_baidu_resources_have_three_fields():
    """Each Baidu resource entry must be a 3-tuple (url, password, size)."""
    for rec, entry in datasets.SEVAR.baidu_resources.items():
        assert len(entry) == 3, (
            f"baidu_resources['{rec}'] should be (url, password, size)"
        )


def test_missing_files_prints_instructions_before_raising():
    """FileNotFoundError must be raised after printing download instructions."""
    tmpdir = tempfile.mkdtemp()
    try:
        with patch.object(datasets.SEVAR, "download_instructions") as mock_instr:
            with pytest.raises(FileNotFoundError):
                datasets.SEVAR(save_to=tmpdir, recording=RECORDING)
            mock_instr.assert_called_once()
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)
