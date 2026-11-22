from pathlib import Path
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application configuration loaded from environment variables or .env file."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    gemini_api_key: str = Field(..., description="Google Gemini API Key")
    gemini_model: str = Field(
        default="gemini-2.5-flash",
        description="Multimodal Gemini model used for page analysis and fallback detection",
    )
    input_dir: Path = Field(default=Path("./input_images"))
    output_pdf: Path = Field(default=Path("./output/compiled_book.pdf"))
    temp_dir: Path = Field(default=Path("./.tmp_processed"))
    enable_scanner_enhancement: bool = Field(
        default=True,
        description="Apply adaptive contrast & sharpening filter to simulate a document scanner",
    )
    min_contour_area_ratio: float = Field(
        default=0.25,
        ge=0.05,
        le=0.95,
        description="Minimum fraction of total image area a detected 4-point contour must cover",
    )
    max_workers: int = Field(
        default=1,
        ge=1,
        le=16,
        description="Concurrent worker threads for image processing & VLM calls",
    )
    log_level: str = Field(default="INFO")


def get_settings() -> Settings:
    return Settings()
