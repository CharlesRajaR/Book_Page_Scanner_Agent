import argparse
import logging
import sys
from pathlib import Path

from src.agent import BookScannerAgent
from src.config import get_settings


def configure_logging(level_name: str) -> None:
    level = getattr(logging, level_name.upper(), logging.INFO)
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Book Page Scanner & PDF Compiler Agent (Computer Vision + Gemini Multimodal)"
    )
    parser.add_argument(
        "-i", "--input",
        type=Path,
        default=None,
        help="Directory containing raw mobile book page photos (default: INPUT_DIR from .env)",
    )
    parser.add_argument(
        "-o", "--output",
        type=Path,
        default=None,
        help="Output PDF file path (default: OUTPUT_PDF from .env)",
    )
    parser.add_argument(
        "--enhance",
        dest="enhance",
        action="store_true",
        default=None,
        help="Enable document scanner contrast & sharpening filter",
    )
    parser.add_argument(
        "--no-enhance",
        dest="enhance",
        action="store_false",
        help="Disable document scanner contrast enhancement",
    )
    parser.add_argument(
        "-w", "--workers",
        type=int,
        default=None,
        help="Number of parallel worker threads",
    )
    parser.add_argument(
        "--keep-temp",
        action="store_true",
        help="Keep intermediate cropped/split page images in TEMP_DIR",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    try:
        settings = get_settings()
    except Exception as exc:
        print(f"Configuration Error: {exc}\nEnsure GEMINI_API_KEY is set in your environment or .env file.", file=sys.stderr)
        return 1

    if args.input is not None:
        settings.input_dir = args.input
    if args.output is not None:
        settings.output_pdf = args.output
    if args.enhance is not None:
        settings.enable_scanner_enhancement = args.enhance
    if args.workers is not None:
        settings.max_workers = args.workers

    configure_logging(settings.log_level)
    logger = logging.getLogger("main")

    try:
        agent = BookScannerAgent(settings)
        output_path = agent.run(keep_temp=args.keep_temp)
        logger.info("Done! Final book PDF saved to: %s", output_path.resolve())
        return 0
    except Exception as exc:
        logger.exception("Pipeline execution failed: %s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())
