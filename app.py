import urllib
from collections import namedtuple
from math import cos, sin
from typing import Any

import gradio as gr
import numpy as np
import rerun as rr
import rerun.blueprint as rrb
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

CUSTOM_PATH = "/"

app = FastAPI()

origins = [
    "https://app.rerun.io",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
)


ColorGrid = namedtuple("ColorGrid", ["positions", "colors"])


def build_color_grid(x_count: int = 10, y_count: int = 10, z_count: int = 10, twist: float = 0) -> ColorGrid:
    """
    Create a cube of points with colors.

    The total point cloud will have x_count * y_count * z_count points.

    Parameters
    ----------
    x_count, y_count, z_count:
        Number of points in each dimension.
    twist:
        Angle to twist from bottom to top of the cube

    """

    grid = np.mgrid[
        slice(-x_count, x_count, x_count * 1j),
        slice(-y_count, y_count, y_count * 1j),
        slice(-z_count, z_count, z_count * 1j),
    ]

    angle = np.linspace(-float(twist) / 2, float(twist) / 2, z_count)
    for z in range(z_count):
        xv, yv, zv = grid[:, :, :, z]
        rot_xv = xv * cos(angle[z]) - yv * sin(angle[z])
        rot_yv = xv * sin(angle[z]) + yv * cos(angle[z])
        grid[:, :, :, z] = [rot_xv, rot_yv, zv]

    positions = np.vstack([xyz.ravel() for xyz in grid])

    colors = np.vstack([
        xyz.ravel()
        for xyz in np.mgrid[
            slice(0, 255, x_count * 1j),
            slice(0, 255, y_count * 1j),
            slice(0, 255, z_count * 1j),
        ]
    ])

    return ColorGrid(positions.T, colors.T.astype(np.uint8))


def html_template(rrd: str, app_url: str = "https://app.rerun.io") -> str:
    encoded_url = urllib.parse.quote(rrd)
    return f"""<div style="width:100%; height:70vh;"><iframe style="width:100%; height:100%;" src="{app_url}?url={encoded_url}" frameborder="0" allowfullscreen=""></iframe></div>"""


def show_cube(x: int, y: int, z: int) -> str:
    rr.init("my data")

    cube = build_color_grid(int(x), int(y), int(z), twist=0)
    rr.log("cube", rr.Points3D(cube.positions, colors=cube.colors, radii=0.5))

    blueprint = rrb.Spatial3DView(origin="cube")

    rr.save("cube.rrd", default_blueprint=blueprint)

    return "cube.rrd"


with gr.Blocks() as demo:
    with gr.Row():
        x_count = gr.Number(minimum=1, maximum=10, value=5, precision=0, label="X Count")
        y_count = gr.Number(minimum=1, maximum=10, value=5, precision=0, label="Y Count")
        z_count = gr.Number(minimum=1, maximum=10, value=5, precision=0, label="Z Count")
        button = gr.Button("Show Cube")
    with gr.Row():
        rrd = gr.File()
    with gr.Row():
        viewer = gr.HTML()

    button.click(show_cube, inputs=[x_count, y_count, z_count], outputs=rrd)
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
