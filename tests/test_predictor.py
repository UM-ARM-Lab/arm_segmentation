import tempfile

import torch
import torchvision
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor, MaskRCNN_ResNet50_FPN_Weights

from arm_segmentation.predictor import Predictor

def test_predictor():
    tmp = tempfile.NamedTemporaryFile(suffix='.pth')
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(weights=MaskRCNN_ResNet50_FPN_Weights.DEFAULT)
    torch.save({'model': model, 'coco': None, 'colors': {}}, tmp.name)
    p = Predictor(tmp.name)


if __name__ == "__main__":
    test_predictor()
