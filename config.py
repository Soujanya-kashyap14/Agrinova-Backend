"""Application configuration loaded from environment variables."""

from functools import lru_cache
from typing import List

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Central settings for EcoAgri Intelligence backend."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ------------------------------------------------------------------
    # Application
    # ------------------------------------------------------------------
    app_name: str = "EcoAgri Intelligence API"
    app_version: str = "1.0.0"
    debug: bool = False

    # ------------------------------------------------------------------
    # Server
    # ------------------------------------------------------------------
    host: str = "0.0.0.0"
    port: int = 8000

    # ------------------------------------------------------------------
    # MongoDB
    # ------------------------------------------------------------------
    mongodb_uri: str = "mongodb://127.0.0.1:27017"
    mongodb_db_name: str = "AgriNova"

    # ------------------------------------------------------------------
    # JWT
    # ------------------------------------------------------------------
    jwt_secret_key: str = "change-me-in-production-use-a-long-random-string"
    jwt_algorithm: str = "HS256"
    jwt_access_token_expire_minutes: int = 60 * 24 * 7  # 7 days
    
        # ------------------------------------------------------------------
    # Gemini AI
    # ------------------------------------------------------------------
    gemini_api_key: str = ""
    # ------------------------------------------------------------------
    # OpenWeather
    # ------------------------------------------------------------------
    openweather_api_key: str = ""
    
    
    
    

    # ------------------------------------------------------------------
    # CORS
    # ------------------------------------------------------------------
    cors_origins: str = (
    "http://localhost:3000,"
    "http://localhost:5173,"
    "http://localhost:8080,"
    "http://127.0.0.1:3000,"
    "http://127.0.0.1:5173,"
    "http://127.0.0.1:8080,"
    "http://10.48.248.159:8080,"
    "http://10.170.32.159:8080"
)
    # ------------------------------------------------------------------
    # Uploads
    # ------------------------------------------------------------------
    upload_dir: str = "uploads"
    max_upload_size_mb: int = 10
    allowed_image_types: str = (
        "image/jpeg,image/png,image/webp,image/jpg"
    )

    @property
    def cors_origins_list(self) -> List[str]:
        return [
            origin.strip()
            for origin in self.cors_origins.split(",")
            if origin.strip()
        ]

    @property
    def allowed_image_types_list(self) -> List[str]:
        return [
            image_type.strip()
            for image_type in self.allowed_image_types.split(",")
            if image_type.strip()
        ]


@lru_cache
def get_settings() -> Settings:
    """Return cached application settings."""
    return Settings()

settings = get_settings()

print("=" * 50)
print("Gemini Key:", settings.gemini_api_key)
print("=" * 50)