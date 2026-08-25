import time
import urllib.request

import torch
from PIL import Image
from torchvision import models

from src.logger import logger


class ImageClassifier:
    def __init__(self) -> None:
        logger.info("Initializing PyTorch MobileNetV3 Model")
        weights = models.MobileNet_V3_Small_Weights.DEFAULT
        self.model = models.mobilenet_v3_small(weights=weights)
        self.model.eval()
        self.preprocess = weights.transforms()

        url = (
            "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
        )
        # decode("utf-8") converts bytes (b'nematode') into a standard string ('nematode')
        with urllib.request.urlopen(url) as response:
            self.labels = [line.decode("utf-8").strip() for line in response.readlines()]
        # self.labels = [line.strip() for line in urllib.request.urlopen(url)]

    def predict(self, image: Image.Image) -> dict[str, float]:
        start_time = time.time()

        img_tensor = self.preprocess(image).unsqueeze(0)
        with torch.no_grad():
            output = self.model(img_tensor)

        probabilities = torch.nn.functional.softmax(output[0], dim=0)
        top_prob, top_catid = torch.topk(probabilities, 1)

        latency = time.time() - start_time

        class_name = self.labels[top_catid[0].item()]
        confidence = float(top_prob[0].item())

        logger.info(
            "prediction_complete",
            class_name=class_name,
            confidence=confidence,
            latency_s=latency,
        )
        return {class_name: confidence}
