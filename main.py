#!/usr/bin/env python3

from __future__ import annotations

import rerun as rr
from datasets import load_dataset

# download/load dataset in pyarrow format
dataset = load_dataset("lerobot/pusht", split="train")

# select the frames belonging to episode number 5
ds_subset = dataset.filter(lambda frame: frame["episode_id"] == 5)

# load all frames in RAM in PIL format
frames = ds_subset["observation.image"]

rr.init("rerun_example_lerobot", spawn=True)

for i, frame in enumerate(frames):
    rr.set_time_sequence("frame", i)
    rr.log("observation/image", rr.Image(frame))
