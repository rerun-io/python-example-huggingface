from pathlib import Path
import urllib
from collections import namedtuple
from math import cos, sin
from typing import Any

import gradio as gr
import numpy as np
from dataset_conversion import log_dataset_to_rerun
import rerun as rr
import rerun.blueprint as rrb
from datasets import load_dataset
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from gradio_huggingfacehub_search import HuggingfaceHubSearch

CUSTOM_PATH = "/"

app = FastAPI()

origins = [
    "https://app.rerun.io",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
)


def html_template(rrd: str, app_url: str = "https://app.rerun.io") -> str:
    encoded_url = urllib.parse.quote(rrd)
    return f"""<div style="width:100%; height:70vh;"><iframe style="width:100%; height:100%;" src="{app_url}?url={encoded_url}" frameborder="0" allowfullscreen=""></iframe></div>"""


def show_dataset(dataset_id: str, episode_id: int) -> str:
    rr.init("dataset")

    # TODO(jleibs): manage cache better and put in proper storage
    filename = Path(f"tmp/{dataset_id}_{episode_id}.rrd")
    if not filename.exists():
        filename.parent.mkdir(parents=True, exist_ok=True)

        rr.save(filename.as_posix())

        dataset = load_dataset(dataset_id, split="train", streaming=True)

        # This is for LeRobot datasets (https://huggingface.co/lerobot):
        ds_subset = dataset.filter(lambda frame: "episode_index" not in frame or frame["episode_index"] == episode_id)

        log_dataset_to_rerun(ds_subset)

    return filename.as_posix()


with gr.Blocks() as demo:
    with gr.Row():
        search_in = HuggingfaceHubSearch(
            "lerobot/pusht",
            label="Search Huggingface Hub",
            placeholder="Search for models on Huggingface",
            search_type="dataset",
        )
        episode_id = gr.Number(1, label="Episode ID")
        button = gr.Button("Show Dataset")
    with gr.Row():
        rrd = gr.File()
    with gr.Row():
        viewer = gr.HTML()

    button.click(show_dataset, inputs=[search_in, episode_id], outputs=rrd)
    rrd.change(
        html_template,
        js="""(rrd) => { console.log(rrd.url); return rrd.url}""",
        inputs=[rrd],
        outputs=viewer,
        preprocess=False,
    )


app = gr.mount_gradio_app(app, demo, path=CUSTOM_PATH)


# Run this from the terminal as you would normally start a FastAPI app: `uvicorn run:app`
# and navigate to http://localhost:8000 in your browser.
