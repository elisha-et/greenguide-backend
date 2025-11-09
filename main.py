from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import base64
import requests
from io import BytesIO
from PIL import Image
import traceback
import json
import random
from config import settings

app = FastAPI(
    title="GreenGuide API",
    description="AI-powered waste classification and disposal advisor",
    version="2.0.0"
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
    """
    Call NVIDIA Nemotron vision model to identify and validate the object
    Returns: dict with is_waste_item, item_name, rejection_reason, confidence
    """
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
                        "text": """You are a waste classification expert. Analyze this image carefully.

First, determine if this is a waste/disposal item that someone would throw away or recycle.

If YES (it's a waste item):
- Identify the specific item (e.g., 'plastic water bottle', 'banana peel', 'AA battery', 'cotton t-shirt')
- Be specific about material when possible (e.g., 'aluminum can' not just 'can')

If NO (not a waste item):
- Determine why: person, animal, landscape, building, multiple_items, unclear, food_on_plate, empty_image

Respond ONLY with JSON in this exact format:
{
  "is_waste_item": true or false,
  "item_name": "specific item name or null",
  "rejection_reason": "reason code or null",
  "confidence": 0.0 to 1.0
}

Examples:
- Plastic bottle → {"is_waste_item": true, "item_name": "plastic water bottle", "rejection_reason": null, "confidence": 0.95}
- Person's face → {"is_waste_item": false, "item_name": null, "rejection_reason": "person", "confidence": 0.98}
- Blurry image → {"is_waste_item": false, "item_name": null, "rejection_reason": "unclear", "confidence": 0.3}"""
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
        "max_tokens": 150,
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
        content = result["choices"][0]["message"]["content"].strip()
        
        # Try to parse JSON response
        try:
            # Remove markdown code blocks if present
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                content = content.split("```")[1].split("```")[0].strip()
            
            parsed = json.loads(content)
            print(f"   ✅ Vision result: {parsed}")
            return parsed
            
        except json.JSONDecodeError:
            # Fallback: try to extract info from text
            print(f"   ⚠️ Could not parse JSON, using fallback")
            return {
                "is_waste_item": True,
                "item_name": content.lower(),
                "rejection_reason": None,
                "confidence": 0.7
            }
    
    except requests.exceptions.Timeout:
        print("   ❌ Request timed out")
        raise HTTPException(status_code=504, detail="Vision model request timed out")
    except requests.exceptions.RequestException as e:
        print(f"   ❌ Request error: {e}")
        raise HTTPException(status_code=500, detail=f"Vision model error: {str(e)}")
    except Exception as e:
        print(f"   ❌ Unexpected error: {e}")
        raise HTTPException(status_code=500, detail=f"Vision processing error: {str(e)}")


def call_reasoning_model(object_name):
    """
    Call NVIDIA Llama-Nemotron reasoning model to determine disposal category
    Returns: dict with category, preparation_steps, confidence
    """
    headers = {
        "Authorization": f"Bearer {settings.NVIDIA_API_KEY}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": settings.REASONING_MODEL,
        "messages": [
            {
                "role": "system",
                "content": """You are a waste management expert. Classify items into these categories:

1. recyclable - Clean paper, cardboard, glass bottles/jars, metal cans, plastic bottles/containers (#1-7)
2. compostable - Food scraps, fruit/vegetable peels, coffee grounds, yard waste, paper napkins
3. landfill - Contaminated items, mixed materials, chip bags, styrofoam, used napkins
4. hazardous - Batteries, paint, chemicals, motor oil, pesticides, fluorescent bulbs
5. e-waste - Electronics, phones, computers, cables, small appliances, chargers
6. textile - Clothes, shoes, bags, fabric, towels, bedding

Provide 2-3 brief preparation steps if needed."""
            },
            {
                "role": "user",
                "content": f"""Classify: {object_name}

Respond ONLY with JSON:
{{
  "category": "one of: recyclable, compostable, landfill, hazardous, e-waste, textile",
  "preparation_steps": ["step 1", "step 2"],
  "confidence": 0.0 to 1.0
}}"""
            }
        ],
        "max_tokens": 200,
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
        content = result["choices"][0]["message"]["content"].strip()
        
        # Parse JSON response
        try:
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                content = content.split("```")[1].split("```")[0].strip()
            
            parsed = json.loads(content)
            
            # Validate category
            valid_categories = ["recyclable", "compostable", "landfill", "hazardous", "e-waste", "textile"]
            if parsed.get("category") not in valid_categories:
                # Fallback to landfill if invalid
                parsed["category"] = "landfill"
                parsed["confidence"] = 0.5
            
            print(f"   ✅ Category: {parsed['category']} (confidence: {parsed.get('confidence', 0.8)})")
            return parsed
            
        except json.JSONDecodeError:
            # Fallback parsing
            content_lower = content.lower()
            category = "landfill"
            
            if "recyclable" in content_lower or "recycle" in content_lower:
                category = "recyclable"
            elif "compost" in content_lower:
                category = "compostable"
            elif "hazardous" in content_lower:
                category = "hazardous"
            elif "e-waste" in content_lower or "electronic" in content_lower:
                category = "e-waste"
            elif "textile" in content_lower or "fabric" in content_lower:
                category = "textile"
            
            return {
                "category": category,
                "preparation_steps": [],
                "confidence": 0.7
            }
    
    except requests.exceptions.Timeout:
        print("   ❌ Request timed out")
        raise HTTPException(status_code=504, detail="Reasoning model request timed out")
    except requests.exceptions.RequestException as e:
        print(f"   ❌ Request error: {e}")
        raise HTTPException(status_code=500, detail=f"Reasoning model error: {str(e)}")


def call_educator_model(object_name, category):
    """
    Call NVIDIA Nemotron educator model to provide environmental impact feedback
    Returns: dict with primary_metric and feedback text
    """
    headers = {
        "Authorization": f"Bearer {settings.NVIDIA_API_KEY}",
        "Content-Type": "application/json"
    }
    
    # Randomly select a primary metric to focus on
    metric_type = random.choice(settings.IMPACT_METRICS)
    
    # Map metric types to better prompts
    metric_prompts = {
        "co2_savings": "CO2 emissions prevented",
        "energy_savings": "energy saved (use relatable comparisons like 'power a laptop for X hours')",
        "water_conservation": "water conserved (in gallons or liters)",
        "resource_conservation": "raw materials saved",
        "landfill_space_saved": "landfill space saved",
        "pollution_reduction": "pollution prevented"
    }
    
    payload = {
        "model": settings.EDUCATOR_MODEL,
        "messages": [
            {
                "role": "system",
                "content": f"""You are an environmental educator. Create engaging, specific feedback about the environmental impact of proper disposal.

Focus on: {metric_prompts[metric_type]}

Requirements:
1. Start with the primary benefit
2. Include a SPECIFIC, QUANTIFIED metric (e.g., "saves enough energy to charge your phone 500 times", not just "saves energy")
3. Add one interesting fact or comparison
4. Keep it under 3 sentences
5. Be encouraging and friendly, not preachy

Make the user feel good about their eco-friendly choice!"""
            },
            {
                "role": "user",
                "content": f"Item: {object_name}\nDisposal: {category}\n\nProvide environmental impact feedback focusing on {metric_type.replace('_', ' ')}."
            }
        ],
        "max_tokens": 200,
        "temperature": 0.7
    }
    
    try:
        print(f"📚 Calling educator model: {settings.EDUCATOR_MODEL}")
        print(f"   Focus metric: {metric_type}")
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
        
        return {
            "primary_metric": metric_type,
            "feedback": feedback
        }
    
    except requests.exceptions.Timeout:
        print("   ❌ Request timed out")
        raise HTTPException(status_code=504, detail="Educator model request timed out")
    except requests.exceptions.RequestException as e:
        print(f"   ❌ Request error: {e}")
        raise HTTPException(status_code=500, detail=f"Educator model error: {str(e)}")


def get_friendly_rejection_message(rejection_reason):
    """Return user-friendly message for rejected images"""
    messages = {
        "person": "I can help identify waste items! Please take a photo of the item you'd like to dispose of, not people.",
        "animal": "That's adorable! 🐕 But I specialize in waste classification. Try photographing the item you want to dispose of.",
        "landscape": "Nice view! But I need a closer photo of a specific item for disposal guidance.",
        "building": "I see a building, but I need to see the specific item you want to dispose of. Try taking a closer photo.",
        "unclear": "I couldn't identify that clearly. Try taking a closer, well-lit photo of the item.",
        "blurry": "The image is too blurry. Please take a clearer photo of the item you'd like to classify.",
        "food_on_plate": "Is this leftover food? If so, food scraps are typically compostable! Scrape leftovers into a compost bin.",
        "empty_image": "I don't see an item to classify. Please take a photo of what you'd like to dispose of.",
        "multiple_items": "I see multiple items! For best results, photograph one item at a time.",
        "default": "I couldn't identify this as a waste item. Please try taking a clearer photo of a single item you want to dispose of."
    }
    
    return messages.get(rejection_reason, messages["default"])


@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "GreenGuide API is running",
        "version": "2.0.0",
        "features": {
            "waste_categories": 6,
            "confidence_scoring": True,
            "invalid_image_detection": True,
            "environmental_metrics": len(settings.IMPACT_METRICS)
        },
        "categories": list(settings.WASTE_CATEGORIES.keys())
    }


@app.get("/health")
async def health_check():
    """Detailed health check"""
    return {
        "status": "healthy",
        "api_key_configured": settings.validate(),
        "models_loaded": True,
        "waste_categories": settings.WASTE_CATEGORIES
    }


@app.post("/classify")
async def classify_waste(file: UploadFile = File(...)):
    """
    Main endpoint: Accepts an image and returns classification, disposal method, and feedback
    """
    print("\n" + "="*60)
    print("🌱 NEW REQUEST RECEIVED (v2.0)")
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
            
            # Convert to RGB if needed
            if image.mode not in ('RGB', 'L'):
                print(f"   Converting from {image.mode} to RGB")
                image = image.convert('RGB')
            
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
        
        # Step 2: Identify and validate object using vision model
        print("\n[1/3] Identifying object...")
        vision_result = call_vision_model(image_base64)
        
        # Check if this is a valid waste item
        if not vision_result.get("is_waste_item", False):
            rejection_reason = vision_result.get("rejection_reason", "default")
            friendly_message = get_friendly_rejection_message(rejection_reason)
            
            print(f"\n❌ REJECTED: {rejection_reason}")
            print("="*60 + "\n")
            
            result = {
                "success": False,
                "is_waste_item": False,
                "rejection_reason": rejection_reason,
                "message": friendly_message,
                "confidence": vision_result.get("confidence", 0.0)
            }
        
            # Print the JSON being sent
            print(f"\n📤 Sending rejection response:")
            print(json.dumps(result, indent=2))
            print("="*60 + "\n")
            
            return result

        object_name = vision_result.get("item_name", "unknown item")
        vision_confidence = vision_result.get("confidence", 0.8)
        
        # Step 3: Determine disposal category using reasoning model
        print("\n[2/3] Determining disposal category...")
        reasoning_result = call_reasoning_model(object_name)
        
        category = reasoning_result.get("category", "landfill")
        preparation_steps = reasoning_result.get("preparation_steps", [])
        reasoning_confidence = reasoning_result.get("confidence", 0.8)
        
        # Step 4: Get environmental feedback using educator model
        print("\n[3/3] Generating environmental feedback...")
        educator_result = call_educator_model(object_name, category)
        
        # Calculate overall confidence (average of vision and reasoning)
        overall_confidence = (vision_confidence + reasoning_confidence) / 2
        
        # Determine confidence level
        if overall_confidence >= settings.CONFIDENCE_HIGH:
            confidence_level = "high"
        elif overall_confidence >= settings.CONFIDENCE_MEDIUM:
            confidence_level = "medium"
        else:
            confidence_level = "low"
        
        # Get category metadata
        category_info = settings.WASTE_CATEGORIES.get(category, {})
        
        # Build response
        result = {
            "success": True,
            "is_waste_item": True,
            "object": object_name,
            "category": category,
            "category_info": {
                "name": category,
                "icon": category_info.get("icon", "🗑️"),
                "color": category_info.get("color", "#FF9500"),
                "description": category_info.get("description", "")
            },
            "preparation_steps": preparation_steps,
            "confidence": {
                "score": round(overall_confidence, 2),
                "level": confidence_level,
                "vision": round(vision_confidence, 2),
                "reasoning": round(reasoning_confidence, 2)
            },
            "environmental_impact": {
                "primary_metric": educator_result["primary_metric"],
                "feedback": educator_result["feedback"]
            }
        }
        
        print("\n" + "="*60)
        print("✅ REQUEST COMPLETED SUCCESSFULLY")
        print("="*60)
        print(f"Object: {object_name}")
        print(f"Category: {category}")
        print(f"Confidence: {confidence_level} ({overall_confidence:.2f})")
        print(f"Metric: {educator_result['primary_metric']}")
        print("="*60 + "\n")
        
        print(f"\n📤 Sending response:")
        print(json.dumps(result, indent=2))

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
    print("🌱 GREENGUIDE BACKEND SERVER v2.0")
    print("="*60)
    print(f"Vision Model: {settings.VISION_MODEL}")
    print(f"Reasoning Model: {settings.REASONING_MODEL}")
    print(f"Educator Model: {settings.EDUCATOR_MODEL}")
    print(f"API Key Configured: {settings.validate()}")
    print(f"Waste Categories: {len(settings.WASTE_CATEGORIES)}")
    print(f"Impact Metrics: {len(settings.IMPACT_METRICS)}")
    print("="*60)
    print(f"Server starting on: {settings.HOST}:{settings.PORT}")
    print("="*60 + "\n")
    uvicorn.run(app, host=settings.HOST, port=settings.PORT)