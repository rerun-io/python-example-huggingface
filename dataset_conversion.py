from __future__ import annotations

import logging
from typing import Any

import numpy as np
import rerun as rr
from PIL import Image
from tqdm import tqdm

logger = logging.getLogger(__name__)


def to_rerun(column_name: str, value: Any) -> Any:
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
    else:
        return rr.TextDocument(str(value))  # Fallback to text


def log_dataset_to_rerun(dataset: Any) -> None:
    # Special time-like columns for LeRobot datasets (https://huggingface.co/datasets/lerobot/):
    TIME_LIKE = {"index", "frame_id", "timestamp"}

    # Ignore these columns (again, LeRobot-specific):
    IGNORE = {"episode_data_index_from", "episode_data_index_to", "episode_id"}

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
            if column_name in TIME_LIKE or column_name in IGNORE:
                continue

            rr.log(column_name, to_rerun(column_name, cell))
