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

    def predict(self, rgb, min_score_threshold=0.40):
        if isinstance(rgb, np.ndarray):
            rgb = torch.from_numpy(rgb)
        with torch.no_grad():
            predictions = self.model([rgb.to(self.device)])[0]

            roboflow_style_predictions = []
            for i in range(len(predictions['labels'])):
                score = predictions['scores'][i]
                label = predictions['labels'][i]
                mask = predictions['masks'][i].cpu().numpy().squeeze()

                if score < min_score_threshold:
                    continue

                pred_dict = {
                    "confidence": score.item(),
                    "class":      self.coco.cats[label.item()]['name'],
                    "mask":       mask,
                }
                roboflow_style_predictions.append(pred_dict)
        return roboflow_style_predictions
