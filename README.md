---
title: Preview Dataset
emoji: 👀
colorFrom: yellow
colorTo: yellow
sdk: docker
app_port: 7860
pinned: false
license: mit
---

# Rerun visualization of HuggingFace datasets
Visualize HuggingFace datasets using [Rerun](https://www.rerun.io/).

Originally built for the LeRobot datasets:

* https://huggingface.co/lerobot
* https://huggingface.co/datasets/lerobot/pusht

https://github.com/rerun-io/python-example-lerobot/assets/1148717/19e9983c-531f-4c48-9b37-37c5cbe1e0bd

Deployed live on hugging-face: https://huggingface.co/spaces/rerun/preview_dataset

## Getting started (native)
Requires Python 3.10 or higher.

```sh
pip install -r requirements.txt
python main.py --dataset lerobot/aloha_sim_insertion_human
```

## Getting started (gradio)
```sh
pip install -r requirements.txt
uvicorn app:app --reload
```
## Example datasets to explore:
* `lerobot/aloha_sim_insertion_human`
* `lerobot/aloha_sim_insertion_scripted`
* `lerobot/aloha_sim_transfer_cube_human`
* `lerobot/aloha_sim_transfer_cube_scripted`
* `lerobot/pusht`
* `lerobot/xarm_lift_medium`
* `nateraw/kitti`
* `sayakpaul/nyu_depth_v2`

## Deploying to HuggingFace

HuggingFace space runs off of the head `main` branch pushed to: https://huggingface.co/spaces/rerun/preview_dataset/tree/main

To update this from the rerun repository, add the HuggingFace repository as an additional remote,
and then push to it.
```sh
git remote add huggingface git@hf.co:spaces/rerun/preview_dataset
git push huggingface main
```

## Note for the maintainer
You can update this repository with the latest changes from https://github.com/rerun-io/rerun_template by running `scripts/template_update.py update --languages python`.
