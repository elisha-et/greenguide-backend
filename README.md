# 🌱 GreenGuide Backend API

<div align="center">
  <img src="https://img.shields.io/badge/Python-3.8+-blue.svg" />
  <img src="https://img.shields.io/badge/FastAPI-0.120+-green.svg" />
  <img src="https://img.shields.io/badge/NVIDIA-AI-76B900.svg" />
</div>

## 🚀 Overview

GreenGuide Backend is a high-performance RESTful API that powers intelligent waste classification using NVIDIA's state-of-the-art AI models. The service processes images, identifies objects, determines proper disposal methods, and provides personalized environmental impact feedback.

**iOS Frontend Repository:** [GreenGuide iOS](https://github.com/elisha-et/greenguide-ios)

**Live Demo:** [https://greenguide-backend.onrender.com](https://greenguide-backend.onrender.com)

## ✨ Key Features

### 🧠 Multi-Model AI Pipeline
- **Vision Model**: NVIDIA Nemotron Nano 12B V2 VL for object identification and validation
- **Reasoning Model**: NVIDIA Llama 3.3 Nemotron Super 49B V1 for disposal categorization
- **Educator Model**: NVIDIA Nemotron Mini 4B for environmental impact feedback

### 🎯 Smart Classification
- **6 Waste Categories**: Recyclable, Compostable, Landfill, Hazardous, E-waste, Textile
- **Invalid Image Detection**: Automatically rejects non-waste items (people, landscapes, unclear images)
- **Confidence Scoring**: Multi-level confidence assessment (vision + reasoning)
- **Preparation Steps**: Context-aware disposal preparation instructions

### 🌍 Environmental Impact
- **6 Impact Metrics**: CO₂ savings, energy conservation, water savings, resource conservation, landfill space, pollution reduction
- **Dynamic Feedback**: Personalized, quantified environmental impact messages
- **Educational Content**: Engaging facts and comparisons to motivate proper disposal

## 🏗️ Architecture

### Tech Stack
- **Framework**: FastAPI 0.120+ (async/await)
- **AI Provider**: NVIDIA AI Foundation Models
- **Image Processing**: Pillow (PIL) + Base64 encoding
- **HTTP Client**: Requests library
- **Server**: Uvicorn ASGI server
- **Deployment**: Render.com (Production)

### API Endpoints

#### `GET /`
Health check with API overview
```json
{
  "status": "GreenGuide API is running",
  "version": "2.0.0",
  "features": {
    "waste_categories": 6,
    "confidence_scoring": true,
    "invalid_image_detection": true,
    "environmental_metrics": 6
  }
}
```

#### `GET /health`
Detailed health status
```json
{
  "status": "healthy",
  "api_key_configured": true,
  "models_loaded": true,
  "waste_categories": {...}
}
```

#### `POST /classify`
Main classification endpoint

**Request:**
```bash
curl -X POST "https://your-backend.com/classify" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@photo.jpg"
```

**Response (Valid Waste Item):**
```json
{
  "success": true,
  "is_waste_item": true,
  "object": "plastic water bottle",
  "category": "recyclable",
  "category_info": {
    "name": "recyclable",
    "icon": "♻️",
    "color": "#34C759",
    "description": "Paper, glass, metals, and certain plastics"
  },
  "preparation_steps": [
    "Empty and rinse the bottle",
    "Remove cap and label if possible",
    "Place in recycling bin"
  ],
  "confidence": {
    "score": 0.92,
    "level": "high",
    "vision": 0.95,
    "reasoning": 0.89
  },
  "environmental_impact": {
    "primary_metric": "energy_savings",
    "feedback": "Recycling this plastic bottle saves enough energy to power a laptop for 3 hours! You're helping reduce petroleum extraction and keeping plastic out of our oceans."
  }
}
```

**Response (Invalid Image):**
```json
{
  "success": false,
  "is_waste_item": false,
  "rejection_reason": "person",
  "message": "I can help identify waste items! Please take a photo of the item you'd like to dispose of, not people.",
  "confidence": 0.98
}
```

## 🚀 Getting Started

### Prerequisites
- Python 3.8 or higher
- NVIDIA API key ([Get yours here](https://build.nvidia.com/))
- pip package manager

### Local Development

1. **Clone the repository**
   ```bash
   git clone https://github.com/elisha-et/greenguide-backend.git
   cd greenguide-backend
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set environment variables**
   ```bash
   # Create .env file
   echo "NVIDIA_API_KEY=your_api_key_here" > .env
   
   # Or export directly
   export NVIDIA_API_KEY="nvapi-xxxxx"
   ```

5. **Run the server**
   ```bash
   python main.py
   ```
   
   Server will start at `http://localhost:8000`

6. **Test the API**
   ```bash
   # Health check
   curl http://localhost:8000/health
   
   # Classify an image
   curl -X POST http://localhost:8000/classify \
     -F "file=@test_image.jpg"
   ```

## ⚙️ Configuration

### Environment Variables

| Variable | Description | Required | Default |
|----------|-------------|----------|---------|
| `NVIDIA_API_KEY` | Your NVIDIA API key | Yes | None |
| `PORT` | Server port | No | 8000 |
| `HOST` | Server host | No | 0.0.0.0 |

### Waste Categories

Configured in `config.py`:
```python
WASTE_CATEGORIES = {
    "recyclable": {"icon": "♻️", "color": "#34C759"},
    "compostable": {"icon": "🌿", "color": "#30B48D"},
    "landfill": {"icon": "🗑️", "color": "#FF9500"},
    "hazardous": {"icon": "⚠️", "color": "#FF3B30"},
    "e-waste": {"icon": "💻", "color": "#5856D6"},
    "textile": {"icon": "👕", "color": "#FF2D55"}
}
```

### Confidence Thresholds

Adjust in `config.py`:
```python
CONFIDENCE_HIGH = 0.85    # High confidence threshold
CONFIDENCE_MEDIUM = 0.65  # Medium confidence threshold
```

## 📊 Performance

### Response Times
- **Average Classification**: 3-5 seconds
- **Image Processing**: < 500ms
- **Model Inference**: 2-4 seconds (NVIDIA API)

### Throughput
- **Concurrent Requests**: Up to 50 (FastAPI async)
- **Image Size Limit**: 10MB (auto-resized to 1024x1024)
- **Timeout**: 90 seconds per request

### Error Handling
- Automatic retry logic for network failures
- Graceful fallback for JSON parsing errors
- Detailed error messages and status codes

## 📈 Monitoring & Logging

### Request Logging
All requests are logged with:
- Timestamp
- Request ID
- File details (name, size, format)
- Processing steps
- Model responses
- Final classification

Example output:
```
============================================================
🌱 NEW REQUEST RECEIVED (v2.0)
============================================================
📸 Processing file: bottle.jpg
   Original size: 2,345,678 bytes
   Format: JPEG, Dimensions: (3024, 4032)
   Resized to: (1024, 1365)
   Final size: 234,567 bytes

[1/3] Identifying object...
🔍 Calling vision model: nvidia/nemotron-nano-12b-v2-vl
   Status: 200
   ✅ Vision result: {"is_waste_item": true, "item_name": "plastic water bottle", "confidence": 0.95}

[2/3] Determining disposal category...
🤔 Calling reasoning model: nvidia/llama-3.3-nemotron-super-49b-v1
   Status: 200
   ✅ Category: recyclable (confidence: 0.89)

[3/3] Generating environmental feedback...
📚 Calling educator model: nvidia/nemotron-mini-4b-instruct
   Focus metric: energy_savings
   Status: 200
   ✅ Feedback generated (156 chars)

============================================================
✅ REQUEST COMPLETED SUCCESSFULLY
============================================================
```

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/NewFeature`)
3. Commit changes (`git commit -m 'Add NewFeature'`)
4. Push to branch (`git push origin feature/NewFeature`)
5. Open a Pull Request

### Development Guidelines
- Add docstrings to functions
- Update README for new features
- Test with multiple image types

## 🐛 Known Issues

- Large images (>10MB) may timeout on slower connections
- Blurry images may result in lower confidence scores
- Multiple objects in one image may confuse the vision model

---

<div align="center">
  Built with 💚 using NVIDIA AI | Making waste management smarter
</div>
