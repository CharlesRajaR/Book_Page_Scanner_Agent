from pathlib import Path
import cv2
import numpy as np
import pytest

from src.models import PageAnalysisResult, ProcessedPageItem
from src.page_sorter import order_pages
from src.pdf_compiler import compile_pages_to_pdf
from src.vision_tools import (
    detect_page_contour_opencv,
    four_point_transform,
    order_points,
    split_two_page_spread,
)


def _make_dummy_item(
    idx: int,
    page_nums: list[int],
    first_snip: str = "",
    last_snip: str = "",
    is_cover: bool = False,
) -> ProcessedPageItem:
    return ProcessedPageItem(
        source_image=Path(f"raw_{idx}.jpg"),
        processed_image=Path(f"proc_{idx}.jpg"),
        original_index=idx,
        analysis=PageAnalysisResult(
            page_numbers=page_nums,
            first_sentence_snippet=first_snip,
            last_sentence_snippet=last_snip,
            is_blank_or_cover=is_cover,
        ),
    )


def test_order_points():
    shuffled = np.array([[500, 600], [100, 100], [100, 600], [500, 100]], dtype="float32")
    ordered = order_points(shuffled)
    assert np.allclose(ordered[0], [100, 100])  # top-left
    assert np.allclose(ordered[1], [500, 100])  # top-right
    assert np.allclose(ordered[2], [500, 600])  # bottom-right
    assert np.allclose(ordered[3], [100, 600])  # bottom-left


def test_opencv_page_detection_and_warp():
    # Create synthetic dark desk background (800x600) with a white quadrilateral page
    canvas = np.full((800, 600, 3), 30, dtype=np.uint8)
    page_quad = np.array([[100, 120], [500, 100], [520, 700], [80, 680]], dtype=np.int32)
    cv2.fillPoly(canvas, [page_quad], (245, 245, 245))

    detected = detect_page_contour_opencv(canvas, min_area_ratio=0.25)
    assert detected is not None
    assert detected.shape == (4, 2)

    warped = four_point_transform(canvas, detected)
    assert warped.shape[0] > 500
    assert warped.shape[1] > 350


def test_split_two_page_spread():
    spread = np.full((600, 1000, 3), 240, dtype=np.uint8)
    # Dark vertical gutter at x=495
    spread[:, 493:498] = 80
    left, right = split_two_page_spread(spread)
    assert 490 <= left.shape[1] <= 500
    assert 500 <= right.shape[1] <= 510


def test_page_ordering_with_unnumbered_continuity():
    p12 = _make_dummy_item(0, [12], first_snip="Chapter begins here.", last_snip="The agent then decides to")
    p10 = _make_dummy_item(1, [10], first_snip="Introduction to agents.", last_snip="End of page ten.")
    unnumbered_13 = _make_dummy_item(
        2, [], first_snip="execute the selected tool immediately.", last_snip="End of unnumbered page."
    )
    cover = _make_dummy_item(0, [], is_cover=True)
    # Adjust indices so cover comes before p10
    cover.original_index = 0
    p10.original_index = 1
    p12.original_index = 2
    unnumbered_13.original_index = 3

    ordered = order_pages([p12, unnumbered_13, p10, cover])
    names = [item.processed_image.name for item in ordered]
    assert names == ["proc_0.jpg", "proc_1.jpg", "proc_2.jpg", "proc_3.jpg"]


def test_pdf_compilation(tmp_path: Path):
    img1 = tmp_path / "p1.jpg"
    img2 = tmp_path / "p2.jpg"
    blank = np.full((400, 300, 3), 255, dtype=np.uint8)
    cv2.imwrite(str(img1), blank)
    cv2.imwrite(str(img2), blank)

    items = [
        ProcessedPageItem(source_image=img1, processed_image=img1, original_index=0),
        ProcessedPageItem(source_image=img2, processed_image=img2, original_index=1),
    ]
    out_pdf = tmp_path / "output.pdf"
    result = compile_pages_to_pdf(items, out_pdf)
    assert result.exists()
    assert result.stat().st_size > 1000
