"""
This script was adapted from the official pytorch object detection tutorial:
https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html
"""
import argparse
from pathlib import Path

from detection.coco_utils import get_coco_dataset
from predictor import Predictor
from viz import viz_predictions

# TODO: move this to a file in the dataset directory
class_name_to_color = {
    'battery':     (1, 0, 0),
    'vacuum_hose': (0, 0.5, 1),
    'vacuum_neck': (0, 0, 1),
    'vacuum_head': (1, 0, 1),
    'robot_feet':  (1, 0.5, 0),
    'robot_hand':  (1, 1, 0),
    'mess':        (0, 1, 0.5),
}


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
        fig, ax = viz_predictions(rgb, predictions, class_name_to_color)
        fig.show()

        # Save the visualization
        fig.savefig(results / f"{idx}.png")


if __name__ == '__main__':
    main()
