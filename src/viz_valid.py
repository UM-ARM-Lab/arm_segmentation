"""
This script was adapted from the official pytorch object detection tutorial:
https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from matplotlib.lines import Line2D

from detection.coco_utils import get_coco_dataset
from predictor import Predictor

class_name_to_color = {
    'battery':     (1, 0, 0),
    'vacuum_hose': (0, 0.5, 1),
    'vacuum_neck': (0, 0, 1),
    'vacuum_head': (1, 0, 1),
    'robot_feet':  (1, 0.5, 0),
    'robot_hand':  (1, 1, 0),
    'mess':        (0, 1, 0.5),
}


def viz_predictions(rgb, predictions):
    rgb_plt = rgb.permute(1, 2, 0)
    fig, ax = plt.subplots()
    ax.imshow(rgb_plt, alpha=0.5)
    for pred in predictions:
        mask = pred['mask']
        class_name = pred['class']
        confidence = pred['confidence']
        color = class_name_to_color[class_name]
        mask_rgb = np.ones_like(rgb_plt) * color
        mask_a = mask[..., None] * confidence * 0.8  # even p(mask)=1 should be slightly transparent
        mask_rgba = np.concatenate((mask_rgb, mask_a), axis=-1)
        ax.imshow(mask_rgba)
    custom_lines = [Line2D([0], [0], color=c, lw=4) for c in class_name_to_color.values()]
    ax.legend(custom_lines, class_name_to_color.keys())

    return fig, ax


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=Path, help="path to dataset")

    args = parser.parse_args()

    pred = Predictor()

    dataset = get_coco_dataset(args.dataset, 'valid')

    results = Path("results")
    results.mkdir(exist_ok=True)

    for idx, (rgb, _) in enumerate(dataset):
        predictions = pred.predict(rgb)
        fig, ax = viz_predictions(rgb, predictions)
        fig.show()

        # Save the visualization
        fig.savefig(results / f"{idx}.png")



if __name__ == '__main__':
    main()
