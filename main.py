#!/usr/bin/env python3

from __future__ import annotations

import rerun as rr
from datasets import load_dataset

# download/load dataset in pyarrow format
print("Loading dataset…")
dataset = load_dataset("lerobot/pusht", split="train")

# select the frames belonging to episode number 5
print("Select specific episode…")
ds_subset = dataset.filter(lambda frame: frame["episode_id"] == 5)

print("Starting Rerun…")
rr.init("rerun_example_lerobot", spawn=True)

print("Logging to Rerun…")
for frame_id, timestamp, image, state, action, next_reward in zip(
    ds_subset["frame_id"],
    ds_subset["timestamp"],
    ds_subset["observation.image"],
    ds_subset["observation.state"],
    ds_subset["action"],
    ds_subset["next.reward"],
):
    rr.set_time_sequence("frame_id", frame_id)
    rr.set_time_seconds("timestamp", timestamp)
    rr.log("observation/image", rr.Image(image))
    rr.log("observation/state", rr.BarChart(state))
    rr.log("observation/action", rr.BarChart(action))
    rr.log("next/reward", rr.Scalar(next_reward))
