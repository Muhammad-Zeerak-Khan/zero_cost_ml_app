import io
import time

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import HTMLResponse
from PIL import Image
from prometheus_fastapi_instrumentator import Instrumentator
from pydantic import BaseModel

from src.logger import logger
from src.model import ImageClassifier


# 1. Pydantic Models for Strict Data Validation
class PredictionData(BaseModel):
    class_name: str
    confidence: float
    latency_seconds: float


class InferenceResponse(BaseModel):
    filename: str
    success: bool
    data: PredictionData


class HealthResponse(BaseModel):
    status: str
    model: str


# 2. FastAPI App Initialization
app = FastAPI(title="Zero-Cost Vision API")
Instrumentator().instrument(app).expose(app)  # Exposes /metrics for Prometheus

classifier = ImageClassifier()


# 3. The Sleek Custom Web UI
@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    return """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Vision ML Classifier</title>
        <script src="https://cdn.tailwindcss.com"></script>
    </head>
    <body class="bg-gray-900 text-white flex flex-col items-center justify-center min-h-screen p-4">
        <div class="max-w-md w-full bg-gray-800 rounded-xl shadow-2xl p-8 border border-gray-700">
            <h2 class="text-2xl font-bold mb-2 text-center">MobileNetV3 Classifier</h2>
            <p class="text-gray-400 text-center mb-6 text-sm">Upload an image for CPU-optimized inference</p>
            
            <input type="file" id="imageInput" accept="image/*" class="block w-full text-sm text-gray-400 file:mr-4 file:py-2 file:px-4 file:rounded-full file:border-0 file:text-sm file:font-semibold file:bg-blue-600 file:text-white hover:file:bg-blue-700 cursor-pointer mb-4"/>
            
            <button onclick="uploadImage()" class="w-full bg-blue-600 hover:bg-blue-500 text-white font-bold py-2 px-4 rounded-lg transition-colors">Classify Image</button>
            
            <div id="result" class="mt-6 hidden p-4 bg-gray-900 rounded-lg border border-gray-700">
                <p class="text-lg">Class: <span id="predClass" class="font-bold text-green-400"></span></p>
                <p>Confidence: <span id="predConf" class="text-gray-300"></span>%</p>
                <p class="text-xs text-gray-500 mt-2">Latency: <span id="predLat"></span>s</p>
            </div>
        </div>

        <script>
            async function uploadImage() {
                const fileInput = document.getElementById('imageInput');
                if (!fileInput.files.length) return alert('Please select an image first.');
                
                const formData = new FormData();
                formData.append('file', fileInput.files[0]);

                const response = await fetch('/predict', { method: 'POST', body: formData });
                const json = await response.json();

                if(response.ok) {
                    document.getElementById('result').classList.remove('hidden');
                    document.getElementById('predClass').innerText = json.data.class_name;
                    document.getElementById('predConf').innerText = (json.data.confidence * 100).toFixed(2);
                    document.getElementById('predLat').innerText = json.data.latency_seconds.toFixed(4);
                } else {
                    alert('Error: ' + json.detail);
                }
            }
        </script>
    </body>
    </html>
    """


# 4. API Endpoints
@app.post("/predict", response_model=InferenceResponse)
async def predict(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400, detail="Invalid file type. Upload a JPEG/PNG."
        )

    start_time = time.time()
    try:
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        prediction_dict = classifier.predict(image)

        # Extract the single key-value pair from the model output
        class_name = list(prediction_dict.keys())[0]
        confidence = prediction_dict[class_name]

        latency = time.time() - start_time

        return InferenceResponse(
            filename=file.filename,
            success=True,
            data=PredictionData(
                class_name=class_name, confidence=confidence, latency_seconds=latency
            ),
        )
    except Exception as e:
        logger.error("inference_failed", error=str(e))
        raise HTTPException(status_code=500, detail="Internal inference error.")


@app.get("/health", response_model=HealthResponse)
def health_check():
    return HealthResponse(status="healthy", model="MobileNetV3-CPU")
