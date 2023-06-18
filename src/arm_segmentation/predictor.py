from typing import List, Union

import numpy as np
import torch


class Predictor:

    def __init__(self):
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

        model_dict = torch.load('model.pth')
        self.model = model_dict['model']
        self.model.to(self.device)
        self.model.eval()

        # The COCO dataset object used at training time, used for mapping from ints to class names
        self.coco = model_dict['coco']
        self.colors = model_dict['colors']

    def predict(self, rgb_np):
        """ Same as predict_torch but assumes numpy input/output """
        rgb = torch.from_numpy(rgb_np / 255.0).permute(2, 0, 1).float()
        predictions = self.predict_torch(rgb)
        for pred in predictions:
            pred['mask'] = pred['mask'].squeeze().cpu().numpy()

        return predictions

    def predict_torch(self, rgb, min_score_threshold=0.40):
        with torch.no_grad():
            predictions = self.model([rgb.to(self.device)])[0]

            roboflow_style_predictions = []
            for i in range(len(predictions['labels'])):
                score = predictions['scores'][i]
                label = predictions['labels'][i]
                mask = predictions['masks'][i]

                if score < min_score_threshold:
                    continue

                pred_dict = {
                    "confidence": score.item(),
                    "class":      self.coco.cats[label.item()]['name'],
                    "mask":       mask,
                }
                roboflow_style_predictions.append(pred_dict)
        return roboflow_style_predictions


def get_combined_mask(predictions, desired_class_names: Union[str, List[str]]):
    """ Combines all masks for a given class name or list of class names """
    if isinstance(desired_class_names, str):
        desired_class_names = [desired_class_names]

    masks = []
    for pred in predictions:
        class_name = pred["class"]
        mask = pred["mask"]

        if class_name in desired_class_names:
            masks.append(mask)

    if len(masks) == 0:
        return None

    combined_mask = np.clip(np.sum(masks, 0), 0, 1)
    return combined_mask
