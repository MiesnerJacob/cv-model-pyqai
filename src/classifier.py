from transformers import AutoFeatureExtractor, ResNetForImageClassification
import numpy as np
from torch.nn import Softmax as softmax


class ImageClassification:
    """
    Use Resnet-50 Loaded from Hugging Face Hub for Image Classification.
    """

    def __init__(self):
        """
        The constructor for ImageClassification class.
        Attributes:
            feature_extractor: model for extracting features from PIL image
            model: image classification model
        """

        self.feature_extractor = AutoFeatureExtractor.from_pretrained("microsoft/resnet-50")
        self.model = ResNetForImageClassification.from_pretrained("microsoft/resnet-50")

    def classify(self, image):
        """
        Classify Images.
        Parameters:
            image (PIL Image): image to be classified.
        Returns:
            predictions (list): list of top 3 class predictions
        """

        inputs = self.feature_extractor(images=image, return_tensors="pt")
        outputs = self.model(**inputs)
        pred_probs = softmax(dim=1)(outputs.logits)

        predictions = []
        for i in range(1, 4):
            predicted_class_idx = np.argsort(np.max(pred_probs.cpu().detach().numpy(), axis=0))[-i]
            predicted_class_pred_prob = float(pred_probs[(0, predicted_class_idx)].detach().numpy())
            predicted_class_name = self.model.config.id2label[predicted_class_idx]
            predictions.append({'Class': predicted_class_name, 'Pred_Prob': predicted_class_pred_prob})

        return predictions
