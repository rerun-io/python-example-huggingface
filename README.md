---
title: Preview Dataset
emoji: 👀
colorFrom: yellow
colorTo: yellow
sdk: gradio
sdk_version: 4.27.0
app_file: app.py
pinned: false
license: mit
---


# Rerun visualization of HuggingFace datasets
Visualize HuggingFace datasets using [Rerun](https://www.rerun.io/).

Originally built for the LeRobot datasets:

* https://huggingface.co/lerobot
* https://huggingface.co/datasets/lerobot/pusht

https://github.com/rerun-io/python-example-lerobot/assets/1148717/19e9983c-531f-4c48-9b37-37c5cbe1e0bd


## Getting started
Requires Python 3.10 or higher.

```sh
pip install -r requirements.txt
python main.py --dataset lerobot/aloha_sim_insertion_human
```

Example datasets to explore:
* `lerobot/aloha_sim_insertion_human`
* `lerobot/aloha_sim_insertion_scripted`
* `lerobot/aloha_sim_transfer_cube_human`
* `lerobot/aloha_sim_transfer_cube_scripted`
* `lerobot/pusht`
* `lerobot/xarm_lift_medium`
* `nateraw/kitti`
* `sayakpaul/nyu_depth_v2`

## Note for the maintainer
You can update this repository with the latest changes from https://github.com/rerun-io/rerun_template by running `scripts/template_update.py update --languages python`.
