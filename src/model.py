import time
import urllib.request

import spaces
import torch
from PIL import Image
from torchvision import models

from src.logger import logger


class ImageClassifier:
    def __init__(self) -> None:
        logger.info(
            "Initializing PyTorch EfficientNetV2-S Model for High-Accuracy Inference"
        )

        # Load state-of-the-art EfficientNetV2 Small weights
        weights = models.EfficientNet_V2_S_Weights.DEFAULT
        self.model = models.efficientnet_v2_s(weights=weights)
        self.model.eval()
        self.preprocess = weights.transforms()

        # Hardware acceleration setup
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        logger.info(f"Model successfully mapped to device: {self.device}")

        url = (
            "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
        )
        with urllib.request.urlopen(url) as response:
            self.labels = [
                line.decode("utf-8").strip() for line in response.readlines()
            ]

    @spaces.GPU  # Dynamically allocates the free ZeroGPU tier when requested
    def predict(self, image: Image.Image) -> dict[str, float]:
        start_time = time.time()

        # Move image tensor to the hardware accelerator device
        img_tensor = self.preprocess(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            output = self.model(img_tensor)

        probabilities = torch.nn.functional.softmax(output[0], dim=0)
        top_prob, top_catid = torch.topk(probabilities, 1)

        latency = time.time() - start_time

        class_name = self.labels[top_catid[0].item()]
        confidence = float(top_prob[0].item())

        logger.info(
            "efficientnet_prediction_complete",
            model="efficientnet_v2_s",
            class_name=class_name,
            confidence=confidence,
            latency_s=latency,
        )
        return {class_name: confidence}
