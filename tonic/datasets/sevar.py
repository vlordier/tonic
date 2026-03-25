import os
from collections.abc import Callable

import numpy as np
from importRosbag.importRosbag import importRosbag

from tonic.dataset import Dataset
from tonic.io import make_structured_array


class SEVAR(Dataset):
    """`SEVAR <https://github.com/sevar-dataset/sevar>`_

    Stereo Event camera dataset for Virtual and Augmented Reality (SEVAR).

    SEVAR is an event-based/spike-based dataset captured with DAVIS346 stereo event cameras
    designed for virtual/augmented reality (VR/AR) applications. It provides head-mounted
    indoor sequences with scenarios including rapid motion and high dynamic range. Beyond the
    primary event streams, the dataset also includes stereo regular camera images and IMU
    measurements, all hardware time-synchronized. Ground truth 6-DoF poses from a Vicon motion
    capture system are available for trajectory evaluation.

    The dataset contains:

    * **Stereo event data** (primary) from DAVIS346 cameras (346×260 pixels), up to 120 dB
    * Stereo regular camera frames at 30 Hz from OV7251 (640×480 pixels)
    * IMU data at 1000 Hz from ICM42688P
    * Ground truth 6-DoF poses from Vicon motion capture at 300 Hz

    .. note::
        **Automatic download is not available** for this dataset because the rosbag files
        (0.6 – 3 GB each) are hosted exclusively on Baidu Pan, which requires a browser-based
        login and does not support programmatic access.

        Call :meth:`SEVAR.download_instructions` to print the full list of download URLs with
        their access passwords, then place each downloaded file at::

            {save_to}/SEVAR/{recording_name}/{recording_name}.bag

        Optional ground-truth files should be placed at::

            {save_to}/SEVAR/{recording_name}/{recording_name}_gt.txt

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
        data_selection (list of str, optional): Subset of data streams to load and return.
                   Must be a list containing one or more of: ``events_left``,
                   ``events_right``, ``images_left``, ``images_right``, ``imu``.
                   Defaults to ``None`` which loads **all** streams (backward-compatible).
                   To work exclusively with the event-based data pass
                   ``["events_left", "events_right"]``.
        transform (callable, optional): A callable of transforms to apply to the data dict.
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

    # All data streams that can be selected via data_selection.
    data_keys = [
        "events_left",
        "events_right",
        "images_left",
        "images_right",
        "imu",
    ]

    # Per-recording Baidu Pan download information scraped from the official README.
    # Automatic download is impossible (Baidu Pan requires browser login); these URLs
    # are provided here so that download_instructions() can print them for the user.
    # Format: recording_name -> (baidu_url, password, approximate_size)
    baidu_resources: dict[str, tuple[str, str, str]] = {
        "AR-normal":   ("https://pan.baidu.com/s/1DcHsmSs8aXCsQ6BRrOQRyQ", "u84d", "2.28 GB"),
        "AR-hdr":      ("https://pan.baidu.com/s/1I4xRrMlMm_O5IFj8Fgggyg",  "nvn4", "3.03 GB"),
        "VR-normal":   ("https://pan.baidu.com/s/1q1bwJ9ZTNo7omWhokHIQGQ",  "hehy", "2.88 GB"),
        "VR-hdr":      ("https://pan.baidu.com/s/1Hy8fkpfCqzUbw3I7nm0n4w",  "tiat", "2.45 GB"),
        "Board-slow":  ("https://pan.baidu.com/s/1XyZxj_i9jrk1cvzLvA-wpQ",  "6n2a", "0.61 GB"),
        "Board-normal":("https://pan.baidu.com/s/1IuT5doBtop6PRvgcB-UwGQ",  "7iih", "1.30 GB"),
        "Board-fast":  ("https://pan.baidu.com/s/1Djfci73R7RvnqDIfev_ygQ",  "fqpi", "2.79 GB"),
        "Board-hdr":   ("https://pan.baidu.com/s/14mzKbcsr6eISLaZdFZiACw",  "nh28", "0.97 GB"),
        "Desk-normal": ("https://pan.baidu.com/s/1Djfci73R7RvnqDIfev_ygQ",  "fqpi", "1.54 GB"),
        "Desk-fast":   ("https://pan.baidu.com/s/1HJva8meORZXoJwYAOkIh5g",  "aw99", "2.69 GB"),
        "Desk-hdr":    ("https://pan.baidu.com/s/1qrsD1o4dksL5qjxY46WweA",  "txyv", "2.37 GB"),
        "Walk-slow":   ("https://pan.baidu.com/s/1AglEb6pitj0bilx0KhxQtQ",  "fngm", "1.16 GB"),
        "Walk-normal": ("https://pan.baidu.com/s/12Zxfvr9ft9tbOARvWh-SkA",  "xueu", "1.30 GB"),
        "Sofa-normal": ("https://pan.baidu.com/s/1v57p_W20rdeRv_EDrZ5zug",  "qhgb", "0.75 GB"),
        "Sofa-fast":   ("https://pan.baidu.com/s/1CwrT_a98gY5l7t7VDJHiWA",  "ekwv", "2.09 GB"),
        "Sofa-hdr":    ("https://pan.baidu.com/s/1E3QR-Ao94SvqBWvyb0bPVQ",  "nxew", "1.01 GB"),
    }

    # Per-recording OneDrive URLs for the optional ground-truth text files.
    onedrive_gt_resources: dict[str, str] = {
        "AR-normal":   "https://1drv.ms/t/c/48c1f55133f3a070/EeWoHTQA1QtCk0tByN716boBsaj-6KRwnviLcxNZXlqWMA?e=7Af0jd",
        "AR-hdr":      "https://1drv.ms/t/c/48c1f55133f3a070/ESrI-6U_KU5Ns54_CNN4jGYBgd_o71BTwBbdNDwN4F9NTg?e=EYrIVi",
        "VR-normal":   "https://1drv.ms/t/c/48c1f55133f3a070/EVhB8s1AlqdFrMVp4o4g_DoB3GqabFZ7rOmprJ3qDWNoXg?e=PcpEOj",
        "VR-hdr":      "https://1drv.ms/t/c/48c1f55133f3a070/EXC0AhGF76VCv6U02hDiFQsBTA0r_LKgpcqHyK5p-zSzvA?e=R3YwF1",
        "Board-slow":  "https://1drv.ms/t/c/48c1f55133f3a070/EcfDi3SwX7RFtJU1RAyO2KMBGwqTe6ntZw3jckGryfDMkg?e=JLeWH0",
        "Board-normal":"https://1drv.ms/t/c/48c1f55133f3a070/EciLEcak0KZJmHXXBwNRMCgB4PNLzpjUmuPwHh5tKTQk-Q?e=loeyWv",
        "Board-fast":  "https://1drv.ms/t/c/48c1f55133f3a070/ESHQVmqu0b1Jun-db3TxzRgBQMDycFducz8UmSlNcPK7xA?e=j2azvV",
        "Board-hdr":   "https://1drv.ms/t/c/48c1f55133f3a070/EUm7wPhhMYpJndDCZdGwbCgBAstKpKeKINYh8EljoWNrsw?e=hJfnTF",
        "Desk-normal": "https://1drv.ms/t/c/48c1f55133f3a070/ESPLFC29HchEquE3akbo_60Brd5q6lID0U9gc8dtVallFg?e=AvBn8s",
        "Desk-fast":   "https://1drv.ms/t/c/48c1f55133f3a070/ESZKLTyAJopNtRwpci9_WCcBbXhZ3UJwFZzOr_a3xb6Ozg?e=Fn8aoM",
        "Desk-hdr":    "https://1drv.ms/t/c/48c1f55133f3a070/EQ3mIkFWfkZJqp2bLpsgQB4BKLUoZ0lsPLSdqQ4Kyr_hzA?e=yZysT9",
        "Walk-slow":   "https://1drv.ms/t/c/48c1f55133f3a070/Ee-qZdqIREBDozHGLSsUI8wB1zCN8sqy0P-7j73Qp_S-Bw?e=PKgh17",
        "Walk-normal": "https://1drv.ms/t/c/48c1f55133f3a070/EaQu1pG_JN5Og9SQpe_dpF0Bgv7fO7PitPotX7k-C-SBfA?e=grLSwq",
        "Sofa-normal": "https://1drv.ms/t/c/48c1f55133f3a070/EY3rW8noaFFOldg6yakm02UBmj9xHw45zF1x2zQCPA9xIQ?e=VZFOYI",
        "Sofa-fast":   "https://1drv.ms/t/c/48c1f55133f3a070/EXOuVL0YV_NMt3ZtfCW949ABErpuFnNBixKqPd8amOvr0Q?e=jQEiT8",
        "Sofa-hdr":    "https://1drv.ms/t/c/48c1f55133f3a070/EYG3hTMpmLZMk2r94t2AnW4B6i1YufTE3us65yDMe8YSSw?e=wniUeT",
    }

    def __init__(
        self,
        save_to: str,
        recording: str | list[str],
        data_selection: list[str] | None = None,
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

        if data_selection is None:
            self.data_selection = list(self.data_keys)
        else:
            data_selection = (
                [data_selection] if isinstance(data_selection, str) else list(data_selection)
            )
            invalid = [k for k in data_selection if k not in self.data_keys]
            if invalid:
                raise ValueError(
                    f"Invalid data_selection key(s): {invalid}. "
                    f"Choose from: {self.data_keys}"
                )
            self.data_selection = data_selection

        if not self._check_exists():
            self.download_instructions(selection=self.selection)
            raise FileNotFoundError(
                f"SEVAR bag files not found in {self.location_on_system}. "
                "See the download instructions printed above. "
                "Place each downloaded .bag file at "
                "{save_to}/SEVAR/{recording_name}/{recording_name}.bag"
            )

    @classmethod
    def download_instructions(cls, selection: list[str] | None = None) -> None:
        """Print manual download instructions for SEVAR recordings.

        Automatic download is not possible because the bag files are hosted on Baidu Pan,
        which requires a browser-based login. This method prints every URL, its access
        password, the approximate file size, and where to place the file on disk.

        Args:
            selection: List of recording names to show instructions for.
                       Defaults to all recordings.
        """
        recs = selection if selection is not None else cls.recordings
        sep = "-" * 78
        print(sep)
        print("SEVAR – Manual Download Instructions")
        print("Source: https://github.com/sevar-dataset/sevar")
        print(sep)
        print(
            "Automatic download is not available: Baidu Pan requires a browser login.\n"
            "Please download each recording manually and save it as:\n"
            "  {save_to}/SEVAR/{recording_name}/{recording_name}.bag\n"
        )
        print(f"{'Recording':<16} {'Size':>8}  {'Password':>8}  Baidu Pan URL")
        print(sep)
        for rec in recs:
            if rec in cls.baidu_resources:
                url, pwd, size = cls.baidu_resources[rec]
                print(f"{rec:<16} {size:>8}  {pwd:>8}  {url}")
        print(sep)
        print("\nOptional ground-truth files (TUM format, ~few KB each) – OneDrive:")
        print("  Place at: {save_to}/SEVAR/{recording_name}/{recording_name}_gt.txt\n")
        for rec in recs:
            if rec in cls.onedrive_gt_resources:
                print(f"  {rec:<16}  {cls.onedrive_gt_resources[rec]}")
        print(sep)

    def __getitem__(self, index):
        """
        Returns:
            tuple of (data, target) where *data* is a dict whose keys are a subset of
            ``events_left``, ``events_right``, ``images_left``, ``images_right``, ``imu``
            as specified by *data_selection* (all streams by default), and *target* is a
            numpy array of shape ``(N, 8)`` with columns
            ``(timestamp, tx, ty, tz, qx, qy, qz, qw)`` or ``None`` when no ground-truth
            file is present.

            Event streams (``events_left`` / ``events_right``) are returned as NumPy
            structured arrays with dtype ``[('x', int), ('y', int), ('t', int), ('p', int)]``
            where ``t`` is in microseconds, zero-referenced to the first event of each stream.
        """
        recording = self.selection[index]
        bag_path = os.path.join(
            self.location_on_system, recording, f"{recording}.bag"
        )

        topics = importRosbag(filePathOrName=bag_path, log="ERROR")

        data = {}

        # --- event-based / spike-based streams (DAVIS346, 346×260) ---
        if "events_left" in self.data_selection:
            raw = dict(topics["/davis/left/events"])
            raw["ts"] = (raw["ts"] - raw["ts"][0]) * 1e6
            data["events_left"] = make_structured_array(
                raw["x"], raw["y"], raw["ts"], raw["pol"], dtype=self.dtype
            )

        if "events_right" in self.data_selection:
            raw = dict(topics["/davis/right/events"])
            raw["ts"] = (raw["ts"] - raw["ts"][0]) * 1e6
            data["events_right"] = make_structured_array(
                raw["x"], raw["y"], raw["ts"], raw["pol"], dtype=self.dtype
            )

        # --- regular-frame streams (OV7251, 640×480, 30 Hz) ---
        if "images_left" in self.data_selection:
            data["images_left"] = np.stack(topics["/camera/left"]["frames"])

        if "images_right" in self.data_selection:
            data["images_right"] = np.stack(topics["/camera/right"]["frames"])

        # --- inertial stream (ICM42688P, 1000 Hz) ---
        if "imu" in self.data_selection:
            imu = dict(topics["/imu/data"])
            imu["ts"] = ((imu["ts"] - imu["ts"][0]) * 1e6).astype(int)
            data["imu"] = imu

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
