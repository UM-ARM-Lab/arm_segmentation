[![Python package](https://github.com/UM-ARM-Lab/arm_segmentation/actions/workflows/python-package.yml/badge.svg)](https://github.com/UM-ARM-Lab/arm_segmentation/actions/workflows/python-package.yml)
[![Upload Python Package](https://github.com/UM-ARM-Lab/arm_segmentation/actions/workflows/python-publish.yml/badge.svg)](https://github.com/UM-ARM-Lab/arm_segmentation/actions/workflows/python-publish.yml)

This repository contains very simple training, inference, and visualization code for instance segmentation.
This is a good choice for when you don't want to get into the details and just want a model that works well and is cheap
and fast (compared to services like AWS)

# Data

This repository assumes you have used RoboFlow to annotate your dataset.
You should export your dataset as a COCO style dataset, which stores annotations in JSON.

# Installation & Setup

## From Pip (recommended)
You can install via pip
```
pip install arm_segmentation
```

Train!
```
python -m arm_segmentation.train ~/path/to/datset.zip # can also be the extracted folder instead of zip
```

## From source

Or you can clone the source code and install it that way.

1. Clone this repository
   ```
   git clone git@github.com:UM-ARM-Lab/arm_segmentation.git
   ```
2. In an existing or new python virtual environment, install the dependencies
    ```
    pip install -r requirements.txt
    ```
3. Train
    ```
    # You may need export PYTHONPATH=./src, or you can try pip install -e .
    ./scripts/train.py path_to_dataset
    ./scripts/visualize_validation.py path_to_dataset
   ```

# Inference

```
    from arm_segmentation.predictor import Predictor
    predictor = Predictor('path/to/model.pth')
    # Assumes rgb_np is a [h, w, 3] numpy array. If you have a torch tensor, use `predict_torch` instead.
    predictions = predictor.predict(rgb_np)
```
