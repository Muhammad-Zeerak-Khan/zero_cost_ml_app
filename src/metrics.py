from prometheus_client import Counter, Histogram

# Track total number of predictions
PREDICTION_REQUESTS = Counter(
    "model_inference_requests_total", "Total number of inference requests"
)

# Track latency of the PyTorch forward pass
INFERENCE_LATENCY = Histogram(
    "model_inference_latency_seconds", "Time spent running the PyTorch model"
)
