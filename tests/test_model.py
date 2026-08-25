import pytest
from PIL import Image

from src.model import ImageClassifier


def test_model_initialization():
    """Test that the PyTorch model and ImageNet labels load correctly."""
    classifier = ImageClassifier()

    assert classifier.model is not None, "PyTorch model failed to initialize."
    assert len(classifier.labels) == 1000, "ImageNet labels did not fetch correctly."


def test_model_prediction():
    """Test that the prediction returns the expected dictionary format."""
    classifier = ImageClassifier()

    # Create a dummy solid black image (224x224) to simulate user upload
    dummy_image = Image.new("RGB", (224, 224), color="black")
    result = classifier.predict(dummy_image)

    # Verify the output is a dictionary mapping a string to a float
    assert isinstance(result, dict), "Prediction output must be a dictionary."

    for class_name, confidence in result.items():
        assert isinstance(class_name, str), "The predicted class name must be a string."
        assert isinstance(confidence, float), "The confidence score must be a float."
        assert 0.0 <= confidence <= 1.0, (
            "Confidence must be a percentage between 0.0 and 1.0."
        )
