import io
import logging
from pathlib import Path

import cv2
import numpy as np
from google import genai
from google.genai import types
from PIL import Image
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from src.models import FallbackBoundingBox, PageAnalysisResult

logger = logging.getLogger(__name__)


def _ndarray_to_pil(bgr_image: np.ndarray) -> Image.Image:
    rgb = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)


def _resize_for_vlm(pil_img: Image.Image, max_dim: int = 1600) -> Image.Image:
    """Downscale large mobile images before sending to VLM to reduce latency and token usage."""
    w, h = pil_img.size
    if max(w, h) <= max_dim:
        return pil_img
    scale = max_dim / float(max(w, h))
    return pil_img.resize((int(w * scale), int(h * scale)), Image.Resampling.LANCZOS)


class GeminiVisionAnalyzer:
    """Multimodal LLM client using Google GenAI SDK with structured Pydantic outputs."""

    def __init__(self, api_key: str, model_name: str = "gemini-2.5-flash"):
        self.client = genai.Client(api_key=api_key)
        self.model_name = model_name

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1.5, min=2, max=15),
        retry=retry_if_exception_type(Exception),
        reraise=True,
    )
    def detect_page_bbox(self, bgr_image: np.ndarray) -> FallbackBoundingBox:
        """
        Fallback Vision tool: asks Gemini to locate the tight bounding box of the book page
        when classical OpenCV contour detection fails (e.g., white page on white desk).
        """
        pil_img = _resize_for_vlm(_ndarray_to_pil(bgr_image), max_dim=1024)

        prompt = (
            "Locate the main book page (or open two-page book spread) in this photograph. "
            "Exclude surrounding table surfaces, hands, fingers, phones, or background objects. "
            "Return the tight bounding box normalized to 0-1000."
        )

        response = self.client.models.generate_content(
            model=self.model_name,
            contents=[pil_img, prompt],
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=FallbackBoundingBox,
                temperature=0.0,
            ),
        )
        return FallbackBoundingBox.model_validate_json(response.text)

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1.5, min=2, max=15),
        retry=retry_if_exception_type(Exception),
        reraise=True,
    )
    def analyze_page(self, bgr_image: np.ndarray) -> PageAnalysisResult:
        """
        Inspect a cropped book page image to extract:
        - Printed page number(s)
        - Whether it is a 2-page spread
        - Needed clockwise rotation (0, 90, 180, 270)
        - First & last body text snippets for sentence continuity ordering
        """
        pil_img = _resize_for_vlm(_ndarray_to_pil(bgr_image), max_dim=1600)

        prompt = (
            "You are a document digitization assistant analyzing a photograph of a book page.
"
            "Inspect the image carefully and extract:
"
            "1. `page_numbers`: All printed page numbers visible in headers/footers, ordered left-to-right. "
            "Do NOT confuse chapter numbers, section numbers, or dates with page numbers. "
            "If no page number is printed, return an empty list [].
"
            "2. `is_two_page_spread`: True ONLY if the photo clearly shows BOTH left and right facing pages "
            "of an open book side-by-side.
"
            "3. `rotation_needed_degrees`: Clockwise rotation (0, 90, 180, or 270) required so the text is right-side up.
"
            "4. `first_sentence_snippet`: The first ~15 words of actual body text at the top of the page "
            "(ignore running headers and page numbers).
"
            "5. `last_sentence_snippet`: The last ~15 words of actual body text at the bottom of the page "
            "(ignore footers and page numbers).
"
            "6. `is_blank_or_cover`: True if this is a cover, title page, blank page, or full-page illustration.
"
            "7. `confidence`: Your confidence (0.0 to 1.0) in the page number and orientation reading."
        )

        response = self.client.models.generate_content(
            model=self.model_name,
            contents=[pil_img, prompt],
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=PageAnalysisResult,
                temperature=0.0,
            ),
        )
        return PageAnalysisResult.model_validate_json(response.text)
