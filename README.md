# Book Page Scanner & PDF Compiler Agent

A production-ready, hybrid **Computer Vision + Multimodal AI Agent** pipeline that takes raw mobile camera photos of book pages, removes background clutter/edges, fixes perspective distortion, detects two-page spreads, intelligently orders pages using **Gemini 2.5 Flash** (via page number extraction & sentence continuity), and compiles them into a single lossless PDF.

---

## Key Features

- **EXIF Normalization:** Automatically fixes mobile camera orientation tags before processing.
- **Hybrid Edge Removal & Perspective Dewarping:**
  1. **Primary (Deterministic CV):** Uses OpenCV Canny edge detection + morphological closing + 4-point contour approximation to locate the book page and warp perspective top-down.
  2. **Fallback (Vision LLM Bounding Box):** If the page is photographed on a low-contrast surface (e.g., white paper on a white table) where OpenCV fails, the agent falls back to Gemini Vision to locate the `[ymin, xmin, ymax, xmax]` bounding box and crops cleanly.
- **Two-Page Spread Splitting:** Automatically detects horizontal two-page spreads and splits them cleanly down the center gutter before ordering.
- **Intelligent Page Ordering:**
  - Uses Gemini Structured Outputs (`Pydantic`) to extract printed page numbers, rotation corrections, and first/last sentence snippets.
  - Sorts numbered pages numerically and slots unnumbered pages (title pages, chapter dividers, illustrations) by matching text continuity across page boundaries.
- **Document Scanner Enhancement (Optional):** Applies CLAHE local contrast enhancement and subtle sharpening for crisp, readable text without destroying illustrations.
- **Lossless PDF Compilation:** Bundles ordered pages using `img2pdf` without re-encoding bloat.

---

## How the Application Works

The pipeline processes every input image through the following stages:

1. **Discover images** — Finds supported image files in `input_images/` and sorts their filenames.
2. **Correct orientation** — Uses EXIF metadata to return mobile photos to the correct orientation.
3. **Remove page edges** — Tries OpenCV contour detection and perspective correction first.
4. **Use the Gemini fallback** — If OpenCV cannot find a page, Gemini receives the image and locates a normalized page bounding box.
5. **Analyze the page** — Gemini detects clockwise rotation, page numbers, text snippets, blank/cover pages, and two-page spreads.
6. **Split spreads** — If one photograph contains two facing pages, the image is divided at the central gutter.
7. **Enhance pages** — Optional CLAHE contrast and sharpening improve readability.
8. **Order pages** — Printed page numbers are sorted numerically; unnumbered pages are placed using capture position and sentence-continuity clues.
9. **Compile PDF** — `img2pdf` places the final images into `output/my_book.pdf` in A4 proportions.

Temporary processed images are removed after a successful run unless `--keep-temp` is used.

---

## Quickstart

### 1. Install Dependencies
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure API Key
```bash
# Create .env manually and add:
GEMINI_API_KEY=your_actual_api_key
GEMINI_MODEL=gemini-2.5-flash
MAX_WORKERS=1
```

### 3. Run the Agent
Place your mobile book photos (`.jpg`, `.jpeg`, `.png`, `.webp`) into `input_images/` and run:
```bash
python main.py --input ./input_images --output ./output/my_book.pdf
```

### CLI Options
```bash
python main.py \
  --input ./input_images \
  --output ./output/my_book.pdf \
  --enhance \
  --workers 1 \
  --keep-temp
```

- `--input / -i`: Directory containing raw mobile book images.
- `--output / -o`: Destination path for the final PDF.
- `--enhance / --no-enhance`: Toggle document scanner contrast enhancement.
- `--workers / -w`: Number of concurrent threads for vision & LLM processing.
- `--keep-temp`: Keep intermediate cropped/split page images for debugging.

---

## Virtual Environment Setup

This project should run inside a virtual environment because modern Linux distributions may use PEP 668 to protect system Python packages.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Stop an active environment with:

```bash
deactivate
```

Do not use `pip install --break-system-packages` for normal project setup. It can modify protected system packages.

---

## API and Quota Behavior

Gemini structured-output requests use the **V1beta API**. The application also limits default concurrent processing to one worker so requests are less likely to reach Gemini's free-tier request quota.

If Gemini is unavailable or has reached a quota limit, the pipeline continues safely:

- Bounding-box failure: the original image is processed without a Gemini crop.
- Page-analysis failure: the page receives a neutral result with zero page-number confidence.
- No printed page numbers: the pages fall back to capture order, with left halves of spreads before right halves.

These fallbacks allow the PDF to be compiled. However, automatic rotation, page-number sorting, and sentence continuity may be less accurate until Gemini is available again.

For a more stable run:

- Keep `MAX_WORKERS=1` on free-tier API limits.
- Wait for the quota reset if the API reports `429 RESOURCE_EXHAUSTED`.
- Verify that the Gemini key is valid and has access to `gemini-2.5-flash`.
- Never place real API keys in source code or commit them to Git.

---

## Common Setup and Runtime Issues

### Missing `cv2` or `google-genai`

This usually means packages were installed into a different Python environment. Activate the project environment before running commands:

```bash
source .venv/bin/activate
python -m pip show opencv-python-headless google-genai
```

### `pytest` cannot import `src`

The included `pytest.ini` sets the project directory as Python's import path:

```ini
[pytest]
pythonpath = .
```

Use either command:

```bash
python -m pytest -v
pytest -v
```

### Gemini reports that JSON mode is unsupported

Structured responses require the V1beta API. `src/llm_analyzer.py` configures the client with:

```python
http_options=types.HttpOptions(api_version="v1beta")
```

### Gemini returns HTTP 429

This means the API quota or requests-per-minute limit was exhausted. The application waits and retries where configured, then uses local fallbacks if the request still fails.

### Incorrect page order

Check the following:

- Capture pages in a consistent reading sequence.
- Ensure printed page numbers are visible and correctly oriented.
- Use `--keep-temp` to inspect the cleaned intermediate pages.
- Add `-w 1` to avoid concurrent Gemini requests.
- Check the logs for `detected_page`, `method`, and final PDF page order.

---

## Important Fixes Applied

- Repaired malformed Python string literals in the Gemini prompt and the page-continuity text comparison.
- Added a project virtual environment instead of changing protected system Python packages.
- Added `pytest.ini` so tests can import the `src` package correctly.
- Corrected test fixture filenames so they match their adjusted capture indices.
- Configured Gemini to use V1beta for Pydantic JSON structured responses.
- Changed the default worker count to `1` to reduce API quota pressure.
- Added local fallbacks for bounding-box detection and page analysis.
- Preserved capture order when no page numbers can be detected.

---

## Security

Keep `.env` private. The `.gitignore` file excludes `.env`, but a key already committed to Git or shared publicly should be revoked and replaced.

Minimum safety checks:

```bash
git status --ignored
```

Never include the real `GEMINI_API_KEY` value in README examples, screenshots, logs, issues, or committed files.

---

## Project Structure

```text
book_page_scanner_agent/
├── src/
│   ├── __init__.py
│   ├── config.py          # Pydantic settings & environment configuration
│   ├── models.py          # Structured data schemas for agent state & VLM outputs
│   ├── vision_tools.py    # OpenCV cropping, perspective warp, splitting & enhancement
│   ├── llm_analyzer.py    # Gemini analyzer, retries & API fallbacks
│   ├── page_sorter.py     # Numerical sorting + sentence-continuity resolver
│   ├── pdf_compiler.py    # Lossless PDF generator via img2pdf
│   └── agent.py           # Main stateful orchestrator
├── tests/
│   └── test_pipeline.py   # Unit tests for CV geometry & page ordering
├── input_images/          # Raw mobile book photos
├── output/                # Generated PDFs
├── .venv/                 # Local Python environment
├── main.py                # CLI entry point
├── pytest.ini             # Makes src importable during test collection
├── requirements.txt       # Python dependencies
├── .env                   # Private API key and local configuration
└── README.md
```
