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

## Project Structure

```text
book_page_scanner_agent/
├── src/
│   ├── __init__.py
│   ├── config.py          # Pydantic settings & environment configuration
│   ├── models.py          # Structured data schemas for agent state & VLM outputs
│   ├── vision_tools.py    # OpenCV cropping, perspective warp, splitting & enhancement
│   ├── llm_analyzer.py    # Gemini 2.5 Flash multimodal analyzer & fallback bbox detector
│   ├── page_sorter.py     # Numerical sorting + sentence-continuity resolver
│   ├── pdf_compiler.py    # Lossless PDF generator via img2pdf
│   └── agent.py           # Main stateful orchestrator with retries & concurrency
├── tests/
│   └── test_pipeline.py   # Unit tests for CV geometry & page ordering logic
├── input_images/          # Place raw mobile photos here
├── output/                # Compiled PDF output directory
├── main.py                # CLI entry point
├── requirements.txt
├── .env.example
└── README.md
```

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
cp .env.example .env
# Edit .env and set GEMINI_API_KEY=your_actual_api_key
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
  --workers 4 \
  --keep-temp
```

- `--input / -i`: Directory containing raw mobile book images.
- `--output / -o`: Destination path for the final PDF.
- `--enhance / --no-enhance`: Toggle document scanner contrast enhancement.
- `--workers / -w`: Number of concurrent threads for vision & LLM processing.
- `--keep-temp`: Keep intermediate cropped/split page images for debugging.

---

## Running Tests
```bash
pytest -v
```
