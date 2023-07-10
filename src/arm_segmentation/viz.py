from typing import Dict

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D


def viz_predictions(rgb, predictions, class_name_to_color: Dict, fig=None, ax=None, legend=True):
    """
    Visualize the predictions on a single image.

    Args:
        rgb: RGB image as a numpy array of shape (3, H, W)
        predictions: list of dicts, each dict has keys 'mask', 'class', 'confidence'
        class_name_to_color: dict mapping class names to RGB colors
        fig: matplotlib figure, if None, a new figure will be created
        ax: matplotlib axis
        legend: whether to show a legend

    Returns:
        fig, ax: matplotlib figure and axis objects
    """
    if rgb.shape[0] == 3:
        rgb = rgb.permute(1, 2, 0)

    if fig is None or ax is None:
        fig, ax = plt.subplots()

    ax.imshow(rgb, alpha=0.5)
    for pred in predictions:
        mask_rgba = viz_prediction(class_name_to_color, pred, rgb)
        ax.imshow(mask_rgba)

    if legend:
        custom_lines = [Line2D([0], [0], color=c, lw=4) for c in class_name_to_color.values()]
        ax.legend(custom_lines, class_name_to_color.keys())

    return fig, ax


def viz_prediction(class_name_to_color, pred, rgb):
    mask = pred['mask']
    class_name = pred['class']
    confidence = pred['confidence']
    color = class_name_to_color[class_name]
    mask_rgb = np.ones_like(rgb) * color
    mask_a = mask[..., None] * confidence * 0.8  # even p(mask)=1 should be slightly transparent
    mask_rgba = np.concatenate((mask_rgb, mask_a), axis=-1)
    return mask_rgba
