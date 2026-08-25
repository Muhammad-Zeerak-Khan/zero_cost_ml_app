import gradio as gr
import uvicorn
from fastapi import FastAPI
from prometheus_client import make_asgi_app

from src.metrics import PREDICTION_REQUESTS
from src.model import ImageClassifier

app = FastAPI(title="MLOps Portfolio App")

# Mount Prometheus metrics endpoint
metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)

classifier = ImageClassifier()


def process_image(image):
    PREDICTION_REQUESTS.inc()
    return classifier.predict(image)


# Build the Gradio UI
with gr.Blocks(title="Zero-Cost Vision API") as demo:
    gr.Markdown("# ImageNet Vision Classifier")
    gr.Markdown(
        "Upload an image. The model runs on PyTorch MobileNetV3. Metrics are exported to Grafana."
    )

    with gr.Row():
        image_input = gr.Image(type="pil", label="Upload Image")
        label_output = gr.Label(num_top_classes=1, label="Prediction")

    submit_btn = gr.Button("Classify")
    submit_btn.click(fn=process_image, inputs=image_input, outputs=label_output)

# Mount Gradio onto the root of the FastAPI app
app = gr.mount_gradio_app(app, demo, path="/")

if __name__ == "__main__":
    # Hugging Face exposes port 7860 by default
    uvicorn.run(app, host="0.0.0.0", port=7860)
