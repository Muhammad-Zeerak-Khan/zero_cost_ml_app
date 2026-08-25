# 🚀 Zero-Cost ML Vision Microservice

A production-grade, highly optimized PyTorch Computer Vision API built for ultra-low latency and $0 cost deployment. 

Designed with enterprise cloud patterns, this microservice leverages a multi-stage Docker build to shrink ML image weight by >60%, utilizing CPU-optimized PyTorch execution via MobileNetV3 for sub-100ms inference.

## 🏗️ Architecture & Stack
* **Frameworks:** FastAPI, PyTorch (MobileNetV3-Small)
* **Observability:** Prometheus, Grafana, Structlog
* **Containerization:** Multi-Stage Docker, Docker Compose
* **Deployment:** Azure Container Apps (Scale-to-Zero), Azure Container Registry

## 🛠️ Local Development (Docker Compose)
Boot the entire ecosystem (API, Metrics Scraper, Grafana Dashboard) locally in seconds:

```bash
docker-compose up --build