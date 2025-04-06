"""
Configuration module for the PDF processing pipeline.
"""
import os
from pydantic import Field
from pydantic_settings import BaseSettings  # Changed import location
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


class Settings(BaseSettings):
    """Application settings."""
    
    # Supabase configuration
    SUPABASE_URL: str = Field(
        default="",
        description="Supabase URL"
    )
    SUPABASE_KEY: str = Field(
        default="", 
        description="Supabase API key"
    )
    
    # OpenAI configuration
    OPENAI_API_KEY: str = Field(
        default="",
        description="OpenAI API key"
    )
    
    # PDF processing configuration
    MIN_IMAGE_SIZE: int = Field(
        default=100,
        description="Minimum width/height for an image to be considered a chart"
    )
    CHART_DETECTION_THRESHOLD: float = Field(
        default=0.7,
        description="Confidence threshold for chart detection"
    )
    PROXIMITY_THRESHOLD: int = Field(
        default=50,
        description="Threshold for spatial proximity (in points)"
    )
    
    # Storage configuration
    TEMP_DIR: str = Field(
        default="./temp",
        description="Directory for temporary files"
    )
    
    class Config:
        """Pydantic configuration."""
        env_file = ".env"
        env_file_encoding = "utf-8"


def get_settings() -> Settings:
    """
    Get application settings.
    
    Returns:
        Settings: Application settings
    """
    return Settings(
        SUPABASE_URL=os.getenv("SUPABASE_URL", ""),
        SUPABASE_KEY=os.getenv("SUPABASE_KEY", ""),
        OPENAI_API_KEY=os.getenv("OPENAI_API_KEY", ""),
        MIN_IMAGE_SIZE=int(os.getenv("MIN_IMAGE_SIZE", "100")),
        CHART_DETECTION_THRESHOLD=float(os.getenv("CHART_DETECTION_THRESHOLD", "0.7")),
        PROXIMITY_THRESHOLD=int(os.getenv("PROXIMITY_THRESHOLD", "50")),
        TEMP_DIR=os.getenv("TEMP_DIR", "./temp")
    )