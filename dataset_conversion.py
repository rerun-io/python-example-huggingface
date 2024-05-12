from __future__ import annotations

import logging
from pathlib import PosixPath
from typing import Any

import cv2
import numpy as np
import rerun as rr
import torch
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from PIL import Image
from tqdm import tqdm

logger = logging.getLogger(__name__)


def get_frame(
    video_path: PosixPath, timestamp: float, video_cache: dict[PosixPath, tuple[np.ndarray, float]] | None = None
) -> np.ndarray:
    """
    Extracts a specific frame from a video.

    `video_path`: path to the video.
    `timestamp`: timestamp of the wanted frame.
    `video_cache`: cache to prevent reading the same video file twice.
    """

    if video_cache is None:
        video_cache = {}
    if video_path not in video_cache:
        cap = cv2.VideoCapture(str(video_path))
        print("new video!")
        frames = []
        while cap.isOpened():
            success, frame = cap.read()
            if success:
                frames.append(frame)
            else:
                break
        frame_rate = cap.get(cv2.CAP_PROP_FPS)
        video_cache[video_path] = (frames, frame_rate)

    frames, frame_rate = video_cache[video_path]
    return frames[int(timestamp * frame_rate)]


def to_rerun(
    column_name: str,
    value: Any,
    video_cache: dict[PosixPath, tuple[np.ndarray, float]] | None = None,
    videos_dir: PosixPath | None = None,
) -> Any:
    """Do our best to interpret the value and convert it to a Rerun-compatible archetype."""
    if isinstance(value, Image.Image):
        if "depth" in column_name:
            return rr.DepthImage(value)
        else:
            return rr.Image(value)
    elif isinstance(value, np.ndarray):
        return rr.Tensor(value)
    elif isinstance(value, list):
        if isinstance(value[0], float):
            return rr.BarChart(value)
        else:
            return rr.TextDocument(str(value))  # Fallback to text
    elif isinstance(value, float) or isinstance(value, int):
        return rr.Scalar(value)
    elif isinstance(value, torch.Tensor):
        if value.dim() == 0:
            return rr.Scalar(value.item())
        elif value.dim() == 1:
            return rr.BarChart(value)
        elif value.dim() == 2 and "depth" in column_name:
            return rr.DepthImage(value)
        elif value.dim() == 2:
            return rr.Image(value)
        elif value.dim() == 3 and (value.shape[2] == 3 or value.shape[2] == 4):
            return rr.Image(value)  # Treat it as a RGB or RGBA image
        else:
            return rr.Tensor(value)
    elif isinstance(value, dict) and "path" in value and "timestamp" in value:
        path = (videos_dir or PosixPath("./")) / PosixPath(value["path"])
        timestamp = value["timestamp"]
        return rr.Image(get_frame(path, timestamp, video_cache=video_cache))
    else:
        return rr.TextDocument(str(value))  # Fallback to text


def log_lerobot_dataset_to_rerun(dataset: LeRobotDataset, episode_index: int) -> None:
    # Special time-like columns for LeRobot datasets (https://huggingface.co/lerobot/):
    TIME_LIKE = {"index", "frame_id", "timestamp"}

    # Ignore these columns (again, LeRobot-specific):
    IGNORE = {"episode_data_index_from", "episode_data_index_to", "episode_id"}

    hf_ds_subset = dataset.hf_dataset.filter(
        lambda frame: "episode_index" not in frame or frame["episode_index"] == episode_index
    )

    video_cache: dict[PosixPath, tuple[np.ndarray, float]] = {}

    for row in tqdm(hf_ds_subset):
        # Handle time-like columns first, since they set a state (time is an index in Rerun):
        for column_name in TIME_LIKE:
            if column_name in row:
                cell = row[column_name]
                if isinstance(cell, torch.Tensor) and cell.dim() == 0:
                    cell = cell.item()
                if isinstance(cell, int):
                    rr.set_time_sequence(column_name, cell)
                elif isinstance(cell, float):
                    rr.set_time_seconds(column_name, cell)  # assume seconds
                else:
                    print(f"Unknown time-like column {column_name} with value {cell}")

        # Now log actual data columns:
        for column_name, cell in row.items():
            if column_name in TIME_LIKE or column_name in IGNORE:
                continue
            else:
                rr.log(
                    column_name,
                    to_rerun(column_name, cell, video_cache=video_cache, videos_dir=dataset.videos_dir.parent),
                )


def log_dataset_to_rerun(dataset: Any) -> None:
    TIME_LIKE = {"index", "frame_id", "timestamp"}

    for row in tqdm(dataset):
        # Handle time-like columns first, since they set a state (time is an index in Rerun):
        for column_name in TIME_LIKE:
            if column_name in row:
                cell = row[column_name]
                if isinstance(cell, int):
                    rr.set_time_sequence(column_name, cell)
                elif isinstance(cell, float):
                    rr.set_time_seconds(column_name, cell)  # assume seconds
                else:
                    print(f"Unknown time-like column {column_name} with value {cell}")

        # Now log actual data columns:
        for column_name, cell in row.items():
            if column_name in TIME_LIKE:
                continue
            rr.log(column_name, to_rerun(column_name, cell))
