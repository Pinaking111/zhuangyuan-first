# SPS GenAI Assignment 2

This repository contains a CNN implementation for CIFAR10 classification with a REST API.

## Requirements

- Docker
- Python 3.12+ (if running locally)

## Quick Start with Docker

1. Build the Docker image:
```bash
docker build -t cifar10-api .
```

2. Run the container:
```bash
docker run -p 8000:8000 cifar10-api
```

3. Access the API at http://localhost:8000/docs

## Local Development

1. Install dependencies:
```bash
pip install uv
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
uv pip install -e .
```

2. Train the model:
```bash
python assignment2/train.py
```

3. Start the API:
```bash
uvicorn assignment2.api:app --reload
```

## API Usage

Send a POST request to `/predict` with an image file to get classification results:

```bash
curl -X POST -F "file=@your_image.jpg" http://localhost:8000/predict
```

Response format:
```json
{
    "label": "predicted_class",
    "confidence": 0.9234
}
```
