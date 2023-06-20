This repository contains very simple training, inference, and visualization code for instance segmentation.
This is a good choice for when you don't want to get into the details and just want a model that works well and is cheap
and fast (compared to services like AWS)

# Data

This repository assumes you have used RoboFlow to annotate your dataset.
You should export your dataset as a COCO style dataset, which stores annotations in JSON.

# Installation & Setup

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
    ./train.py path_to_dataset
    ./visualize_validation.py path_to_dataset
   ```
