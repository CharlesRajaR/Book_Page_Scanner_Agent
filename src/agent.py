import logging
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List

from src.config import Settings
from src.llm_analyzer import GeminiVisionAnalyzer
from src.models import ProcessedPageItem
from src.page_sorter import order_pages
from src.pdf_compiler import compile_pages_to_pdf
from src.vision_tools import (
    crop_with_normalized_bbox,
    detect_page_contour_opencv,
    enhance_document_scan,
    four_point_transform,
    load_normalized_image,
    rotate_image_clockwise,
    save_image,
    split_two_page_spread,
)

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff"}


class BookScannerAgent:
    """
    End-to-end stateful agent orchestrating:
    1. Image discovery & EXIF normalization
    2. Edge removal & perspective correction (OpenCV primary + Gemini VLM fallback)
    3. Orientation correction & two-page spread splitting
    4. Optional document scanner contrast enhancement
    5. Intelligent page ordering (page numbers + text continuity)
    6. Lossless PDF compilation
    """

    def __init__(self, settings: Settings):
        self.settings = settings
        self.analyzer = GeminiVisionAnalyzer(
            api_key=settings.gemini_api_key,
            model_name=settings.gemini_model,
        )

    def discover_images(self, input_dir: Path) -> List[Path]:
        if not input_dir.exists() or not input_dir.is_dir():
            raise FileNotFoundError(f"Input directory does not exist: {input_dir}")

        images = sorted(
            [
                p
                for p in input_dir.iterdir()
                if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
            ],
            key=lambda p: p.name.lower(),
        )
        if not images:
            raise ValueError(f"No supported images found in {input_dir}")
        return images

    def _process_single_raw_image(
        self, image_path: Path, original_index: int, temp_dir: Path
    ) -> List[ProcessedPageItem]:
        """
        Process one raw mobile photograph into 1 (single page) or 2 (two-page spread)
        cleaned, upright, analyzed `ProcessedPageItem` objects.
        """
        logger.info("[%d] Processing raw image: %s", original_index, image_path.name)
        raw_bgr = load_normalized_image(image_path)

        # Stage 1: Edge Removal & Perspective Dewarping
        crop_method = "opencv_perspective"
        corners = detect_page_contour_opencv(
            raw_bgr, min_area_ratio=self.settings.min_contour_area_ratio
        )

        if corners is not None:
            try:
                cropped_bgr = four_point_transform(raw_bgr, corners)
                logger.debug("[%s] Cropped via OpenCV 4-point perspective warp.", image_path.name)
            except Exception as exc:
                logger.warning("[%s] Perspective warp failed (%s); falling back to VLM bbox.", image_path.name, exc)
                corners = None

        if corners is None:
            crop_method = "vlm_bounding_box"
            bbox = self.analyzer.detect_page_bbox(raw_bgr)
            if bbox.detected:
                cropped_bgr = crop_with_normalized_bbox(
                    raw_bgr, bbox.ymin, bbox.xmin, bbox.ymax, bbox.xmax
                )
                logger.debug("[%s] Cropped via Gemini VLM fallback bounding box.", image_path.name)
            else:
                crop_method = "unmodified_fallback"
                cropped_bgr = raw_bgr

        # Stage 2: Initial VLM Layout & Orientation Inspection
        initial_analysis = self.analyzer.analyze_page(cropped_bgr)

        if initial_analysis.rotation_needed_degrees in (90, 180, 270):
            logger.info(
                "[%s] Rotating image clockwise by %d degrees.",
                image_path.name,
                initial_analysis.rotation_needed_degrees,
            )
            cropped_bgr = rotate_image_clockwise(
                cropped_bgr, initial_analysis.rotation_needed_degrees
            )
            # Re-analyze upright image if it was rotated so snippets & spread flags are exact
            initial_analysis = self.analyzer.analyze_page(cropped_bgr)

        # Stage 3: Handle Two-Page Spread vs Single Page
        results: List[ProcessedPageItem] = []

        if initial_analysis.is_two_page_spread:
            logger.info("[%s] Detected two-page spread; splitting into left & right pages.", image_path.name)
            left_bgr, right_bgr = split_two_page_spread(cropped_bgr)

            for side, sub_bgr in [("left", left_bgr), ("right", right_bgr)]:
                if self.settings.enable_scanner_enhancement:
                    sub_bgr = enhance_document_scan(sub_bgr)

                out_name = f"{original_index:04d}_{image_path.stem}_{side}.jpg"
                out_path = save_image(sub_bgr, temp_dir / out_name)

                # Analyze each split page individually for exact page number & sentence boundaries
                sub_analysis = self.analyzer.analyze_page(sub_bgr)
                results.append(
                    ProcessedPageItem(
                        source_image=image_path,
                        processed_image=out_path,
                        original_index=original_index,
                        sub_page_side=side,
                        crop_method=crop_method,
                        analysis=sub_analysis,
                    )
                )
        else:
            if self.settings.enable_scanner_enhancement:
                cropped_bgr = enhance_document_scan(cropped_bgr)

            out_name = f"{original_index:04d}_{image_path.stem}.jpg"
            out_path = save_image(cropped_bgr, temp_dir / out_name)

            results.append(
                ProcessedPageItem(
                    source_image=image_path,
                    processed_image=out_path,
                    original_index=original_index,
                    sub_page_side=None,
                    crop_method=crop_method,
                    analysis=initial_analysis,
                )
            )

        return results

    def run(self, keep_temp: bool = False) -> Path:
        """Execute the full book scanning pipeline and return the generated PDF path."""
        raw_images = self.discover_images(self.settings.input_dir)
        logger.info("Discovered %d raw images in '%s'.", len(raw_images), self.settings.input_dir)

        temp_dir = self.settings.temp_dir
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
        temp_dir.mkdir(parents=True, exist_ok=True)

        all_processed_pages: List[ProcessedPageItem] = []

        try:
            with ThreadPoolExecutor(max_workers=self.settings.max_workers) as executor:
                future_to_path = {
                    executor.submit(self._process_single_raw_image, img_path, idx, temp_dir): img_path
                    for idx, img_path in enumerate(raw_images)
                }

                for future in as_completed(future_to_path):
                    img_path = future_to_path[future]
                    try:
                        items = future.result()
                        all_processed_pages.extend(items)
                    except Exception as exc:
                        logger.error("Failed processing '%s': %s", img_path.name, exc, exc_info=True)
                        raise

            # Stage 4: Order all extracted single pages
            ordered_pages = order_pages(all_processed_pages)

            logger.info("--- Final Page Order ---")
            for rank, item in enumerate(ordered_pages, start=1):
                p_nums = item.analysis.page_numbers if item.analysis else []
                logger.info(
                    "PDF Page %02d <- %s (detected_page=%s, method=%s)",
                    rank,
                    item.processed_image.name,
                    p_nums if p_nums else "unnumbered",
                    item.crop_method,
                )

            # Stage 5: Aggregate into final PDF
            final_pdf = compile_pages_to_pdf(ordered_pages, self.settings.output_pdf)
            return final_pdf

        finally:
            if not keep_temp and temp_dir.exists():
                shutil.rmtree(temp_dir, ignore_errors=True)
