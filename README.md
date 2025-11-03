# GreenGuide Backend

AI-powered waste classification backend using FastAPI and NVIDIA models.

## Features
- Object identification using computer vision
- Smart disposal recommendations (recycle/compost/landfill)
- Environmental impact feedback
- REST API for iOS app integration

## Tech Stack
- FastAPI
- NVIDIA AI Models (Nemotron)
- Python 3.8+

## Local Development

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Set environment variable:
```bash
export NVIDIA_API_KEY=your_key_here
```

3. Run server:
```bash
python main.py
```

Server runs on http://localhost:8000

## API Endpoints

- `GET /` - Health check
- `GET /health` - Detailed health status
- `POST /classify` - Classify waste item (accepts image file)

## Environment Variables

- `NVIDIA_API_KEY` - Your NVIDIA API key (required)
- `PORT` - Server port (default: 8000)