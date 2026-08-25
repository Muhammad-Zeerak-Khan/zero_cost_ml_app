import time
import urllib.request

import torch
from PIL import Image
from torchvision import models

from src.logger import logger


class ImageClassifier:
    def __init__(self) -> None:
        logger.info("Initializing PyTorch MobileNetV3 Model (CPU-Optimized)")

        # Load lightweight MobileNetV3 weights optimized for CPU inference
        weights = models.MobileNet_V3_Small_Weights.DEFAULT
        self.model = models.mobilenet_v3_small(weights=weights)
        self.model.eval()
        self.preprocess = weights.transforms()

        # Force CPU execution
        self.device = torch.device("cpu")
        self.model.to(self.device)
        logger.info("Model successfully mapped to CPU device.")

        url = (
            "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
        )
        with urllib.request.urlopen(url) as response:
            self.labels = [
                line.decode("utf-8").strip() for line in response.readlines()
            ]

    def predict(self, image: Image.Image) -> dict[str, float]:
        start_time = time.time()

        # Process image tensor on CPU
        img_tensor = self.preprocess(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            output = self.model(img_tensor)

        probabilities = torch.nn.functional.softmax(output[0], dim=0)
        top_prob, top_catid = torch.topk(probabilities, 1)

        latency = time.time() - start_time

        class_name = self.labels[top_catid[0].item()]
        confidence = float(top_prob[0].item())

        logger.info(
            "cpu_prediction_complete",
            model="mobilenet_v3_small",
            class_name=class_name,
            confidence=confidence,
            latency_s=latency,
        )
        return {class_name: confidence}
