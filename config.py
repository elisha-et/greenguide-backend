import os
from typing import Optional

class Settings:
    """Application settings and configuration"""
    
    # Get API key from environment variable, fallback to hardcoded for local dev
    NVIDIA_API_KEY: str = os.getenv(
        "NVIDIA_API_KEY"
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
    
    @classmethod
    def validate(cls) -> bool:
        """Check if required settings are configured"""
        return bool(cls.NVIDIA_API_KEY and cls.NVIDIA_API_KEY.startswith("nvapi-"))


settings = Settings()