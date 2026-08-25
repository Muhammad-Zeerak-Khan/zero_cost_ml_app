import gradio as gr

from src.logger import logger
from src.model import ImageClassifier

classifier = ImageClassifier()


def process_image(image):
    logger.info("request_received")
    return classifier.predict(image)


with gr.Blocks(title="Zero-Cost Vision API") as demo:
    gr.Markdown("# ImageNet Vision Classifier")
    gr.Markdown(
        "Upload an image. The model runs on PyTorch MobileNetV3 via the free Hugging Face Gradio SDK."
    )

    with gr.Row():
        image_input = gr.Image(type="pil", label="Upload Image")
        label_output = gr.Label(num_top_classes=1, label="Prediction")

    submit_btn = gr.Button("Classify")
    submit_btn.click(fn=process_image, inputs=image_input, outputs=label_output)

# The Gradio SDK on Hugging Face looks for the 'demo' object to launch automatically.
if __name__ == "__main__":
    demo.launch()
