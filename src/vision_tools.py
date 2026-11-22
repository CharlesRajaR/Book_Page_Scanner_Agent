import logging
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
from PIL import Image, ImageOps

logger = logging.getLogger(__name__)


def load_normalized_image(image_path: Path) -> np.ndarray:
    """
    Load an image via Pillow to honor mobile EXIF orientation tags,
    then convert to an OpenCV BGR numpy array.
    """
    with Image.open(image_path) as pil_img:
        pil_img = ImageOps.exif_transpose(pil_img)
        rgb_img = pil_img.convert("RGB")
        bgr_img = cv2.cvtColor(np.array(rgb_img), cv2.COLOR_RGB2BGR)
    return bgr_img


def order_points(pts: np.ndarray) -> np.ndarray:
    """
    Order 4 corner coordinates into:
    [top-left, top-right, bottom-right, bottom-left].
    """
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]  # top-left has smallest sum
    rect[2] = pts[np.argmax(s)]  # bottom-right has largest sum

    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]  # top-right has smallest (y - x)
    rect[3] = pts[np.argmax(diff)]  # bottom-left has largest (y - x)
    return rect


def four_point_transform(image: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Apply perspective transform to obtain a top-down rectangular view of the page."""
    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    width_bottom = np.linalg.norm(br - bl)
    width_top = np.linalg.norm(tr - tl)
    max_width = max(int(width_bottom), int(width_top))

    height_right = np.linalg.norm(tr - br)
    height_left = np.linalg.norm(tl - bl)
    max_height = max(int(height_right), int(height_left))

    if max_width < 50 or max_height < 50:
        raise ValueError("Detected page dimensions are too small after warp.")

    dst = np.array(
        [
            [0, 0],
            [max_width - 1, 0],
            [max_width - 1, max_height - 1],
            [0, max_height - 1],
        ],
        dtype="float32",
    )

    matrix = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(
        image,
        matrix,
        (max_width, max_height),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    )
    return warped


def detect_page_contour_opencv(
    image: np.ndarray, min_area_ratio: float = 0.25
) -> Optional[np.ndarray]:
    """
    Detect the 4 corners of a book page against surrounding background clutter.
    Uses multi-threshold Canny + morphological closing so text lines inside the page
    merge into a solid region rather than breaking the outer contour.
    """
    orig_h, orig_w = image.shape[:2]
    target_h = 600.0
    ratio = orig_h / target_h
    resized_w = max(1, int(orig_w / ratio))
    small = cv2.resize(image, (resized_w, int(target_h)), interpolation=cv2.INTER_AREA)
    small_area = float(small.shape[0] * small.shape[1])

    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    # Bilateral filter preserves strong page edges while smoothing paper texture
    filtered = cv2.bilateralFilter(gray, 9, 75, 75)

    # Morphological closing suppresses internal text lines
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    closed = cv2.morphologyEx(filtered, cv2.MORPH_CLOSE, kernel, iterations=2)

    for low, high in [(50, 150), (30, 100), (75, 200)]:
        edged = cv2.Canny(closed, low, high)
        edged = cv2.dilate(edged, np.ones((3, 3), np.uint8), iterations=1)

        contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]

        for contour in contours:
            area = cv2.contourArea(contour)
            if area / small_area < min_area_ratio:
                continue

            peri = cv2.arcLength(contour, True)
            for eps in (0.02, 0.03, 0.04):
                approx = cv2.approxPolyDP(contour, eps * peri, True)
                if len(approx) == 4 and cv2.isContourConvex(approx):
                    pts = approx.reshape(4, 2).astype("float32") * ratio
                    return pts

    return None


def crop_with_normalized_bbox(
    image: np.ndarray, ymin: int, xmin: int, ymax: int, xmax: int, pad_ratio: float = 0.005
) -> np.ndarray:
    """Crop an image using normalized [0-1000] coordinates returned by a Vision LLM."""
    h, w = image.shape[:2]
    pad_y = int(h * pad_ratio)
    pad_x = int(w * pad_ratio)

    top = max(0, int((ymin / 1000.0) * h) - pad_y)
    left = max(0, int((xmin / 1000.0) * w) - pad_x)
    bottom = min(h, int((ymax / 1000.0) * h) + pad_y)
    right = min(w, int((xmax / 1000.0) * w) + pad_x)

    if bottom - top < 50 or right - left < 50:
        logger.warning("Fallback bounding box too small; returning full image.")
        return image

    return image[top:bottom, left:right]


def rotate_image_clockwise(image: np.ndarray, degrees: int) -> np.ndarray:
    """Rotate image clockwise by 0, 90, 180, or 270 degrees."""
    deg = degrees % 360
    if deg == 90:
        return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    if deg == 180:
        return cv2.rotate(image, cv2.ROTATE_180)
    if deg == 270:
        return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return image


def split_two_page_spread(image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Split a two-page open book spread into (left_page, right_page).
    Searches near the horizontal center (45%-55% width) for the darkest vertical
    gutter shadow line to split cleanly along the book spine.
    """
    h, w = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    search_start = int(w * 0.45)
    search_end = int(w * 0.55)
    center_band = gray[:, search_start:search_end]

    # Column-wise mean intensity; the spine gutter is typically slightly darker
    col_means = center_band.mean(axis=0)
    gutter_offset = int(np.argmin(col_means))
    split_x = search_start + gutter_offset

    # Fallback if the image is completely uniform
    if split_x <= int(w * 0.40) or split_x >= int(w * 0.60):
        split_x = w // 2

    left_page = image[:, :split_x]
    right_page = image[:, split_x:]
    return left_page, right_page


def enhance_document_scan(image: np.ndarray) -> np.ndarray:
    """
    Apply gentle 'document scanner' enhancement:
    - Converts to LAB color space and applies CLAHE to the L (luminance) channel
      to equalize uneven mobile lighting/shadows while preserving natural colors.
    - Applies mild unsharp masking for crisp text edges.
    """
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l_channel)

    merged_lab = cv2.merge((cl, a_channel, b_channel))
    enhanced = cv2.cvtColor(merged_lab, cv2.COLOR_LAB2BGR)

    # Mild unsharp mask
    gaussian = cv2.GaussianBlur(enhanced, (0, 0), sigmaX=1.5)
    sharpened = cv2.addWeighted(enhanced, 1.25, gaussian, -0.25, 0)
    return sharpened


def save_image(image: np.ndarray, output_path: Path, quality: int = 95) -> Path:
    """Save BGR image to disk as high-quality JPEG."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), image, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    return output_path
