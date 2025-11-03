from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import base64
import requests
from io import BytesIO
from PIL import Image
import traceback
from config import settings

app = FastAPI(
    title="GreenGuide API",
    description="AI-powered waste classification and disposal advisor",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def encode_image_to_base64(image_bytes):
    """Convert image bytes to base64 string"""
    return base64.b64encode(image_bytes).decode('utf-8')


def call_vision_model(image_base64):
    """Call NVIDIA Nemotron vision model to identify the object"""
    headers = {
        "Authorization": f"Bearer {settings.NVIDIA_API_KEY}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": settings.VISION_MODEL,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Identify this object in one or two words. Only respond with the object name, nothing else."
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{image_base64}"
                        }
                    }
                ]
            }
        ],
        "max_tokens": 50,
        "temperature": 0.2
    }
    
    try:
        print(f"🔍 Calling vision model: {settings.VISION_MODEL}")
        response = requests.post(settings.API_ENDPOINT, headers=headers, json=payload, timeout=60)
        
        print(f"   Status: {response.status_code}")
        
        if response.status_code != 200:
            print(f"   Error Response: {response.text}")
            raise HTTPException(
                status_code=response.status_code,
                detail=f"Vision model error: {response.text}"
            )
        
        result = response.json()
        object_name = result["choices"][0]["message"]["content"].strip()
        print(f"   ✅ Identified: {object_name}")
        return object_name
    
    except requests.exceptions.Timeout:
        print("   ❌ Request timed out")
        raise HTTPException(status_code=504, detail="Vision model request timed out")
    except requests.exceptions.RequestException as e:
        print(f"   ❌ Request error: {e}")
        raise HTTPException(status_code=500, detail=f"Vision model error: {str(e)}")
    except (KeyError, IndexError) as e:
        print(f"   ❌ Response parsing error: {e}")
        raise HTTPException(status_code=500, detail=f"Unexpected response format from vision model")


def call_reasoning_model(object_name):
    """Call NVIDIA Llama-Nemotron reasoning model to determine disposal method"""
    headers = {
        "Authorization": f"Bearer {settings.NVIDIA_API_KEY}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": settings.REASONING_MODEL,
        "messages": [
            {
                "role": "user",
                "content": f"For a {object_name}, should it be recycled, composted, or sent to landfill? Answer with only ONE word: recycle, compost, or landfill."
            }
        ],
        "max_tokens": 10,
        "temperature": 0.1
    }
    
    try:
        print(f"🤔 Calling reasoning model: {settings.REASONING_MODEL}")
        response = requests.post(settings.API_ENDPOINT, headers=headers, json=payload, timeout=60)
        
        print(f"   Status: {response.status_code}")
        
        if response.status_code != 200:
            print(f"   Error Response: {response.text}")
            raise HTTPException(
                status_code=response.status_code,
                detail=f"Reasoning model error: {response.text}"
            )
        
        result = response.json()
        disposal_method = result["choices"][0]["message"]["content"].strip().lower()
        
        if "recycle" in disposal_method:
            disposal_method = "recycle"
        elif "compost" in disposal_method:
            disposal_method = "compost"
        else:
            disposal_method = "landfill"
        
        print(f"   ✅ Disposal method: {disposal_method}")
        return disposal_method
    
    except requests.exceptions.Timeout:
        print("   ❌ Request timed out")
        raise HTTPException(status_code=504, detail="Reasoning model request timed out")
    except requests.exceptions.RequestException as e:
        print(f"   ❌ Request error: {e}")
        raise HTTPException(status_code=500, detail=f"Reasoning model error: {str(e)}")
    except (KeyError, IndexError) as e:
        print(f"   ❌ Response parsing error: {e}")
        raise HTTPException(status_code=500, detail=f"Unexpected response format from reasoning model")


def call_educator_model(object_name, disposal_method):
    """Call NVIDIA Nemotron educator model to provide environmental impact feedback"""
    headers = {
        "Authorization": f"Bearer {settings.NVIDIA_API_KEY}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": settings.EDUCATOR_MODEL,
        "messages": [
            {
                "role": "user",
                "content": f"In 2-3 sentences, explain the environmental benefit of choosing to {disposal_method} a {object_name}. Include an estimate of CO2 savings if applicable."
            }
        ],
        "max_tokens": 150,
        "temperature": 0.7
    }
    
    try:
        print(f"📚 Calling educator model: {settings.EDUCATOR_MODEL}")
        response = requests.post(settings.API_ENDPOINT, headers=headers, json=payload, timeout=60)
        
        print(f"   Status: {response.status_code}")
        
        if response.status_code != 200:
            print(f"   Error Response: {response.text}")
            raise HTTPException(
                status_code=response.status_code,
                detail=f"Educator model error: {response.text}"
            )
        
        result = response.json()
        feedback = result["choices"][0]["message"]["content"].strip()
        print(f"   ✅ Feedback generated ({len(feedback)} chars)")
        return feedback
    
    except requests.exceptions.Timeout:
        print("   ❌ Request timed out")
        raise HTTPException(status_code=504, detail="Educator model request timed out")
    except requests.exceptions.RequestException as e:
        print(f"   ❌ Request error: {e}")
        raise HTTPException(status_code=500, detail=f"Educator model error: {str(e)}")
    except (KeyError, IndexError) as e:
        print(f"   ❌ Response parsing error: {e}")
        raise HTTPException(status_code=500, detail=f"Unexpected response format from educator model")


@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "GreenGuide API is running",
        "version": "1.0.0",
        "models": {
            "vision": settings.VISION_MODEL,
            "reasoning": settings.REASONING_MODEL,
            "educator": settings.EDUCATOR_MODEL
        }
    }


@app.get("/health")
async def health_check():
    """Detailed health check"""
    return {
        "status": "healthy",
        "api_key_configured": settings.validate(),
        "models_loaded": True
    }


@app.post("/classify")
async def classify_waste(file: UploadFile = File(...)):
    """
    Main endpoint: Accepts an image and returns classification, disposal method, and feedback
    """
    print("\n" + "="*60)
    print("🌱 NEW REQUEST RECEIVED")
    print("="*60)
    
    try:
        # Read and validate image
        print(f"📸 Processing file: {file.filename}")
        image_bytes = await file.read()
        print(f"   Original size: {len(image_bytes):,} bytes")
        
        # Verify it's a valid image
        try:
            image = Image.open(BytesIO(image_bytes))
            print(f"   Format: {image.format}, Dimensions: {image.size}")
            
            # Resize if image is too large
            max_size = (1024, 1024)
            original_size = image.size
            image.thumbnail(max_size, Image.Resampling.LANCZOS)
            
            if image.size != original_size:
                print(f"   Resized to: {image.size}")
            
            # Convert back to bytes
            output = BytesIO()
            image.save(output, format='JPEG', quality=85)
            image_bytes = output.getvalue()
            print(f"   Final size: {len(image_bytes):,} bytes")
            
        except Exception as e:
            print(f"   ❌ Image validation failed: {e}")
            raise HTTPException(status_code=400, detail=f"Invalid image file: {str(e)}")
        
        # Step 1: Encode image
        print("\n📦 Encoding image to base64...")
        image_base64 = encode_image_to_base64(image_bytes)
        print(f"   Base64 length: {len(image_base64):,} chars")
        
        # Step 2: Identify object using vision model
        print("\n[1/3] Identifying object...")
        object_name = call_vision_model(image_base64)
        
        # Step 3: Determine disposal method using reasoning model
        print("\n[2/3] Determining disposal method...")
        disposal_method = call_reasoning_model(object_name)
        
        # Step 4: Get environmental feedback using educator model
        print("\n[3/3] Generating environmental feedback...")
        feedback = call_educator_model(object_name, disposal_method)
        
        # Return all results
        result = {
            "success": True,
            "object": object_name,
            "disposal_method": disposal_method,
            "environmental_feedback": feedback
        }
        
        print("\n" + "="*60)
        print("✅ REQUEST COMPLETED SUCCESSFULLY")
        print("="*60)
        print(f"Object: {object_name}")
        print(f"Disposal: {disposal_method}")
        print(f"Feedback: {feedback[:100]}...")
        print("="*60 + "\n")
        
        return result
    
    except HTTPException as he:
        print(f"\n❌ HTTP Exception: {he.status_code} - {he.detail}")
        print("="*60 + "\n")
        raise he
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        print(f"Traceback:\n{traceback.format_exc()}")
        print("="*60 + "\n")
        raise HTTPException(status_code=500, detail=f"Server error: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    print("\n" + "="*60)
    print("🌱 GREENGUIDE BACKEND SERVER")
    print("="*60)
    print(f"Vision Model: {settings.VISION_MODEL}")
    print(f"Reasoning Model: {settings.REASONING_MODEL}")
    print(f"Educator Model: {settings.EDUCATOR_MODEL}")
    print(f"API Key Configured: {settings.validate()}")
    print("="*60)
    print(f"Server starting on: {settings.HOST}:{settings.PORT}")
    print("="*60 + "\n")
    uvicorn.run(app, host=settings.HOST, port=settings.PORT)