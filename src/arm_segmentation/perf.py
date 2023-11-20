#!/usr/bin/env python
""" This script can be used to benchmark your GPU """

import torch
from timeit import timeit
import torch.utils.data
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor, MaskRCNN_ResNet50_FPN_Weights

from arm_segmentation.predictor import Predictor
from detection import utils


def main():
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    print(f'Using device: {device}')

    training_dt = timeit(lambda: fake_train(device), number=10)
    print(f"Training time: {training_dt:.5f}")

    inference_dt = timeit(lambda: fake_inference(device), number=10)
    print(f"Inference time: {inference_dt:.5f}")


def fake_inference(device):
    model = make_fake_model(device)
    model.eval()

    for _ in range(10):
        images = [torch.randn(3, 224, 224, device=device)]
        predictions = model(images)


def fake_train(device):
    model = make_fake_model(device)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
    for _ in range(10):
        images = [torch.randn(3, 224, 224, device=device)]
        targets = [{
            'boxes':  torch.tensor([[0, 0, 10, 10], [10, 10, 20, 20]], device=device),
            'labels': torch.tensor([1, 2], device=device),
            'masks':  torch.randn(2, 224, 224, device=device),
        }]
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()


def make_fake_model(device):
    num_classes = 10
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(weights=MaskRCNN_ResNet50_FPN_Weights.DEFAULT)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, hidden_layer, num_classes)
    model.to(device)
    return model


if __name__ == '__main__':
    main()
