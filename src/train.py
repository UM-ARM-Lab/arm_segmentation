"""
This script was adapted from the official pytorch object detection tutorial:
https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html
"""
import argparse
from pathlib import Path

import torch
import torch.utils.data
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor, MaskRCNN_ResNet50_FPN_Weights

from detection.coco_utils import get_coco_dataset
from detection.engine import train_one_epoch
from detection.utils import collate_fn


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=Path, help="path to dataset")

    args = parser.parse_args()

    dataset = get_coco_dataset(args.dataset, 'train')

    torch.manual_seed(1)

    data_loader = torch.utils.data.DataLoader(dataset, batch_size=2, shuffle=True, num_workers=4,
                                              collate_fn=collate_fn)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    num_classes = 8
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(weights=MaskRCNN_ResNet50_FPN_Weights.DEFAULT)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, hidden_layer, num_classes)
    model.to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)

    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    num_epochs = 10
    for epoch in range(num_epochs):
        train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10)
        lr_scheduler.step()
        # save at each epoch, overwriting the previous checkpoint
        torch.save({'model': model, 'coco': dataset.coco}, 'model.pth')


if __name__ == '__main__':
    main()
