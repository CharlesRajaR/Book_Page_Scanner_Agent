import logging
from pathlib import Path
from typing import List

import img2pdf

from src.models import ProcessedPageItem

logger = logging.getLogger(__name__)


def compile_pages_to_pdf(ordered_pages: List[ProcessedPageItem], output_pdf_path: Path) -> Path:
    """
    Aggregate ordered single-page images into a single lossless PDF file using img2pdf.
    """
    if not ordered_pages:
        raise ValueError("Cannot compile PDF: no processed pages provided.")

    output_pdf_path.parent.mkdir(parents=True, exist_ok=True)
    image_paths = [str(item.processed_image) for item in ordered_pages]

    logger.info("Compiling %d ordered pages into '%s'...", len(image_paths), output_pdf_path)

    # Fit each page cleanly onto standard A4/Letter proportions while preserving aspect ratio
    layout_fun = img2pdf.get_layout_fun(
        pagesize=(img2pdf.mm_to_pt(210), img2pdf.mm_to_pt(297)),
        fit=img2pdf.FitMode.into,
    )

    pdf_bytes = img2pdf.convert(image_paths, layout_fun=layout_fun)
    with open(output_pdf_path, "wb") as f:
        f.write(pdf_bytes)

    logger.info("Successfully generated PDF: %s (%.2f MB)", output_pdf_path, len(pdf_bytes) / (1024 * 1024))
    return output_pdf_path
