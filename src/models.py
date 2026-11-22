from pathlib import Path
from typing import Optional
from pydantic import BaseModel, Field


class PageAnalysisResult(BaseModel):
    """Structured output returned by the Vision LLM for a single page or spread."""

    page_numbers: list[int] = Field(
        default_factory=list,
        description=(
            "Printed page numbers visible on the page(s), ordered left-to-right. "
            "Empty list if no page number is printed."
        ),
    )
    is_two_page_spread: bool = Field(
        default=False,
        description="True if the image captures both left and right facing pages of an open book.",
    )
    rotation_needed_degrees: int = Field(
        default=0,
        description="Clockwise rotation in degrees (0, 90, 180, or 270) needed so text is upright.",
    )
    first_sentence_snippet: str = Field(
        default="",
        description="First ~15 words of body text at the top of the page (excluding running headers).",
    )
    last_sentence_snippet: str = Field(
        default="",
        description="Last ~15 words of body text at the bottom of the page (excluding footers).",
    )
    is_blank_or_cover: bool = Field(
        default=False,
        description="True if this is a title page, chapter divider, cover, or illustration without body text.",
    )
    confidence: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Confidence score (0.0 to 1.0) in the extracted page number and orientation.",
    )


class FallbackBoundingBox(BaseModel):
    """Normalized [0-1000] bounding box of the book page when OpenCV contour detection fails."""

    ymin: int = Field(..., ge=0, le=1000, description="Top edge normalized (0-1000)")
    xmin: int = Field(..., ge=0, le=1000, description="Left edge normalized (0-1000)")
    ymax: int = Field(..., ge=0, le=1000, description="Bottom edge normalized (0-1000)")
    xmax: int = Field(..., ge=0, le=1000, description="Right edge normalized (0-1000)")
    detected: bool = Field(
        default=True,
        description="False if no book page can be distinguished in the image.",
    )


class ProcessedPageItem(BaseModel):
    """Internal pipeline artifact representing a single cleaned, single-page image ready for sorting."""

    source_image: Path
    processed_image: Path
    original_index: int
    sub_page_side: Optional[str] = None  # 'left', 'right', or None
    crop_method: str = "opencv_perspective"
    analysis: Optional[PageAnalysisResult] = None
    assigned_sort_key: float = 0.0
