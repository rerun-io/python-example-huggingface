#!/usr/bin/env python3

from __future__ import annotations

import argparse
import logging

import rerun as rr
from datasets import load_dataset

from dataset_conversion import log_dataset_to_rerun

logger = logging.getLogger(__name__)


def main() -> None:
    # Ensure the logging gets written to stderr:
    logging.getLogger().addHandler(logging.StreamHandler())
    logging.getLogger().setLevel(logging.INFO)

    parser = argparse.ArgumentParser(description="Log a HuggingFace dataset to Rerun.")
    parser.add_argument("--dataset", default="lerobot/pusht", help="The name of the dataset to load")
    parser.add_argument("--episode-id", default=1, help="Which episode to select")
    args = parser.parse_args()

    print("Loading dataset…")
    dataset = load_dataset(args.dataset, split="train", streaming=True)

    # This is for LeRobot datasets (https://huggingface.co/lerobot):
    ds_subset = dataset.filter(lambda frame: "episode_index" not in frame or frame["episode_index"] == args.episode_id)

    print("Starting Rerun…")
    rr.init(f"rerun_example_huggingface {args.dataset}", spawn=True)

    print("Logging to Rerun…")
    log_dataset_to_rerun(ds_subset)


if __name__ == "__main__":
    main()
