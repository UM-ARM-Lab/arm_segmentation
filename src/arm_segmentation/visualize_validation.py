#!/usr/bin/env python
"""
This script was adapted from the official pytorch object detection tutorial:
https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html
"""
import argparse
from pathlib import Path

from detection.coco_utils import get_coco_dataset
from arm_segmentation.predictor import Predictor
from arm_segmentation.viz import viz_predictions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=Path, help="path to dataset")
    parser.add_argument("model_path", type=Path, help="path to .pth file")
    args = parser.parse_args()

    pred = Predictor(args.model_path)

    args.dataset = args.dataset.expanduser()
    dataset = get_coco_dataset(args.dataset, 'valid')

    results = Path("results")
    results.mkdir(exist_ok=True)

    for idx, (rgb, _) in enumerate(dataset):
        predictions = pred.predict_torch(rgb)  # Torch inputs and outputs by default
        pred.to_np_inplace(predictions)  # you can explicitly convert to numpy if you want
        # Or if rgb is a numpy image, you can simply use pred.predict(rgb)
        fig, ax = viz_predictions(rgb, predictions, pred.colors)
        fig.show()

        # Save the visualization
        fig.savefig(results / f"{idx}.png")


if __name__ == '__main__':
    main()
