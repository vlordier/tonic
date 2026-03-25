import os
from collections.abc import Callable

import numpy as np
from importRosbag.importRosbag import importRosbag

from tonic.dataset import Dataset
from tonic.io import make_structured_array


class SEVAR(Dataset):
    """`SEVAR <https://github.com/sevar-dataset/sevar>`_

    Stereo Event camera dataset for Virtual and Augmented Reality (SEVAR).

    SEVAR is an event camera dataset designed for virtual/augmented reality (VR/AR) applications.
    It provides head-mounted indoor sequences with scenarios including rapid motion and high
    dynamic range. The dataset includes stereo event camera data, stereo regular camera images,
    and IMU measurements, all hardware time-synchronized. Ground truth 6-DoF poses from a
    Vicon motion capture system are available for trajectory evaluation.

    The dataset contains:

    * Stereo event data from DAVIS346 cameras (346×260 pixels), up to 120 dB dynamic range
    * Stereo regular camera frames at 30 Hz from OV7251 (640×480 pixels)
    * IMU data at 1000 Hz from ICM42688P
    * Ground truth 6-DoF poses from Vicon motion capture at 300 Hz

    .. note:: This dataset requires manual download. Please download the rosbag files from
              https://github.com/sevar-dataset/sevar and place them in the expected directory
              structure: ``{save_to}/SEVAR/{recording_name}/{recording_name}.bag``.
              Ground truth files (optional) should be placed at
              ``{save_to}/SEVAR/{recording_name}/{recording_name}_gt.txt``.

    ::

        @misc{sevar2024,
          title={SEVAR: A Stereo Event Camera Dataset for Virtual and Augmented Reality},
          author={SEVAR Dataset Authors},
          howpublished={\\url{https://github.com/sevar-dataset/sevar}},
          year={2024}
        }

    Parameters:
        save_to (string): Location to save files to on disk.
        recording (string or list): Name of a recording or list of recordings to load.
                   Available recordings: AR-normal, AR-hdr, VR-normal, VR-hdr, Board-slow,
                   Board-normal, Board-fast, Board-hdr, Desk-normal, Desk-fast, Desk-hdr,
                   Walk-slow, Walk-normal, Sofa-normal, Sofa-fast, Sofa-hdr.
                   Use 'all' to load all recordings.
        transform (callable, optional): A callable of transforms to apply to event data.
        target_transform (callable, optional): A callable of transforms to apply to the
                                               targets/labels.
        transforms (callable, optional): A callable of transforms that is applied to both data
                                         and labels at the same time.
    """

    sensor_size = (346, 260, 2)
    dtype = np.dtype([("x", int), ("y", int), ("t", int), ("p", int)])
    ordering = dtype.names

    recordings = [
        "AR-normal",
        "AR-hdr",
        "VR-normal",
        "VR-hdr",
        "Board-slow",
        "Board-normal",
        "Board-fast",
        "Board-hdr",
        "Desk-normal",
        "Desk-fast",
        "Desk-hdr",
        "Walk-slow",
        "Walk-normal",
        "Sofa-normal",
        "Sofa-fast",
        "Sofa-hdr",
    ]

    def __init__(
        self,
        save_to: str,
        recording: str | list[str],
        transform: Callable | None = None,
        target_transform: Callable | None = None,
        transforms: Callable | None = None,
    ):
        super().__init__(
            save_to,
            transform=transform,
            target_transform=target_transform,
            transforms=transforms,
        )

        if recording == "all":
            self.selection = self.recordings
        else:
            self.selection = recording if isinstance(recording, list) else [recording]
            for rec in self.selection:
                if rec not in self.recordings:
                    raise RuntimeError(
                        f"Recording '{rec}' is not available. "
                        f"Please select from: {self.recordings}"
                    )

        if not self._check_exists():
            raise FileNotFoundError(
                f"SEVAR dataset not found in {self.location_on_system}. "
                "Please download the data manually from https://github.com/sevar-dataset/sevar "
                "and place rosbag files in the expected directory structure: "
                "{save_to}/SEVAR/{recording_name}/{recording_name}.bag"
            )

    def __getitem__(self, index):
        """
        Returns:
            tuple of (data, target), where data is a dict containing events_left, events_right,
            images_left, images_right, and imu, and target is a numpy array of ground truth
            poses with columns (timestamp, tx, ty, tz, qx, qy, qz, qw) or None if not
            available.
        """
        recording = self.selection[index]
        bag_path = os.path.join(
            self.location_on_system, recording, f"{recording}.bag"
        )

        topics = importRosbag(filePathOrName=bag_path, log="ERROR")

        # Stereo event cameras (DAVIS346, 346x260)
        events_left_raw = topics["/davis/left/events"]
        events_left_raw["ts"] -= events_left_raw["ts"][0]
        events_left_raw["ts"] *= 1e6
        events_left = make_structured_array(
            events_left_raw["x"],
            events_left_raw["y"],
            events_left_raw["ts"],
            events_left_raw["pol"],
            dtype=self.dtype,
        )

        events_right_raw = topics["/davis/right/events"]
        events_right_raw["ts"] -= events_right_raw["ts"][0]
        events_right_raw["ts"] *= 1e6
        events_right = make_structured_array(
            events_right_raw["x"],
            events_right_raw["y"],
            events_right_raw["ts"],
            events_right_raw["pol"],
            dtype=self.dtype,
        )

        # Stereo regular cameras (OV7251, 640x480, 30 Hz)
        images_left = topics["/camera/left"]
        images_left = np.stack(images_left["frames"])

        images_right = topics["/camera/right"]
        images_right = np.stack(images_right["frames"])

        # IMU (ICM42688P, 1000 Hz)
        imu = topics["/imu/data"]
        imu["ts"] = ((imu["ts"] - imu["ts"][0]) * 1e6).astype(int)

        data = {
            "events_left": events_left,
            "events_right": events_right,
            "images_left": images_left,
            "images_right": images_right,
            "imu": imu,
        }

        # Load ground truth poses if available
        gt_path = os.path.join(
            self.location_on_system, recording, f"{recording}_gt.txt"
        )
        targets = self._load_ground_truth(gt_path)

        if self.transform is not None:
            data = self.transform(data)
        if self.target_transform is not None:
            targets = self.target_transform(targets)
        if self.transforms is not None:
            data, targets = self.transforms(data, targets)
        return data, targets

    def __len__(self):
        return len(self.selection)

    def _load_ground_truth(self, gt_path: str) -> np.ndarray | None:
        """Load ground truth poses from a text file.

        The format is expected to be TUM trajectory format:
        ``timestamp tx ty tz qx qy qz qw``
        Lines starting with '#' are treated as comments.

        Returns:
            numpy array of shape (N, 8) or None if the file is not found.
        """
        if not os.path.isfile(gt_path):
            return None
        poses = []
        with open(gt_path) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                poses.append([float(v) for v in line.split()])
        return np.array(poses) if poses else None

    def _check_exists(self) -> bool:
        return all(
            os.path.isfile(
                os.path.join(self.location_on_system, rec, f"{rec}.bag")
            )
            for rec in self.selection
        )
