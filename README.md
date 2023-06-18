This repository contains the docker file and configuration files for training and inference of instance segmentation.
It is based on [MMDetection](https://mmdetection.readthedocs.io/)
This is a good choice for when you don't want to get into the details and just want a model that works well and is cheap and fast (compared to services like AWS)

# Data

This repository assumes you have used RoboFlow to annotate your dataset.
You should export your dataset as a COCO style dataset, which stores annotations in JSON.

# Installation & Setup

In order to avoid changing system dependencies, we use Docker.
The Dockerfile is based on the official one provided by MMDetection.

1. Install Docker. If you can run `docker compose --version` without errors, then Docker is likely installed correctly.
2. Clone this repository
	```
	git clone git@github.com:UM-ARM-Lab/arm_segmentation.git
	```
3. Build the docker image
	```
    # from arm_segmentation/
    docker compose build base  # this may take a long time, but only needs to be done once.
	```
4. Train
    ```
    docker compose run train
   ```

# Datasets

Make sure you set the environment variable `DETECTRON2_DATASETS` to point to the parent directory of your dataset.
The name/path of the dataset you want to use, along with everything else you might want to change, is in the .yaml config file.
you should append "_train" and "_val" to your dataset names in the .yaml file.

# Training

To train:

```
docker compose run train
```
