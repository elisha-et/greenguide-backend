import os
from typing import Optional

class Settings:
    """Application settings and configuration"""
    
    # Get API key from environment variable, fallback to hardcoded for local dev
    NVIDIA_API_KEY: str = os.getenv(
        "NVIDIA_API_KEY",
        "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"
    )
    
    # API endpoint
    API_ENDPOINT: str = "https://integrate.api.nvidia.com/v1/chat/completions"
    
    # Model names
    VISION_MODEL: str = "nvidia/nemotron-nano-12b-v2-vl"
    REASONING_MODEL: str = "nvidia/llama-3.3-nemotron-super-49b-v1"
    EDUCATOR_MODEL: str = "nvidia/nemotron-mini-4b-instruct"
    
    # Server settings
    HOST: str = "0.0.0.0"
    PORT: int = int(os.getenv("PORT", "8000"))
    
    # CORS settings
    CORS_ORIGINS: list = ["*"]  # In production, specify your iOS app domain
    
    # Waste categories configuration
    WASTE_CATEGORIES = {
        "recyclable": {
            "icon": "♻️",
            "color": "#34C759",
            "description": "Paper, glass, metals, and certain plastics"
        },
        "compostable": {
            "icon": "🌿",
            "color": "#30B48D",
            "description": "Food scraps, yard waste, organic materials"
        },
        "landfill": {
            "icon": "🗑️",
            "color": "#FF9500",
            "description": "Non-recyclable, non-hazardous waste"
        },
        "hazardous": {
            "icon": "⚠️",
            "color": "#FF3B30",
            "description": "Batteries, chemicals, paint, toxic materials"
        },
        "e-waste": {
            "icon": "💻",
            "color": "#5856D6",
            "description": "Electronics, cables, appliances"
        },
        "textile": {
            "icon": "👕",
            "color": "#FF2D55",
            "description": "Clothes, fabric, shoes"
        }
    }
    
    # Environmental impact metrics
    IMPACT_METRICS = [
        "co2_savings",
        "energy_savings",
        "water_conservation",
        "resource_conservation",
        "landfill_space_saved",
        "pollution_reduction"
    ]
    
    # Confidence thresholds
    CONFIDENCE_HIGH = 0.85
    CONFIDENCE_MEDIUM = 0.65
    
    @classmethod
    def validate(cls) -> bool:
        """Check if required settings are configured"""
        return bool(cls.NVIDIA_API_KEY and cls.NVIDIA_API_KEY.startswith("nvapi-"))


settings = Settings()