"""Joke Extraction CLI — process one .eml file and emit JSON to stdout."""

import json
import logging
import os
import sys
import traceback
from datetime import datetime, timezone

from config import LOG_FILE, LOG_LEVEL
from llm import ExtractionResult, extract_joke
from preprocess import PreprocessResult, preprocess


# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

def _setup_logging() -> logging.Logger:
    os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
    logger = logging.getLogger("joke_extractor")
    logger.setLevel(getattr(logging, LOG_LEVEL, logging.INFO))

    if not logger.handlers:
        handler = logging.FileHandler(LOG_FILE, encoding="utf-8")
        handler.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)s %(message)s", datefmt='%Y-%m-%d %H:%M:%S')
        )
        logger.addHandler(handler)

    return logger


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

def _emit(data: dict) -> None:
    """Print JSON to stdout."""
    print(json.dumps(data, ensure_ascii=False))


def _failure(message: str) -> dict:
    return {"status": "failure", "message": message}


def _success(prep: PreprocessResult, result: ExtractionResult) -> dict:
    return {
        "status": "success",
        "from": prep.from_addr,
        "title": result.title,
        "title_source": result.title_source,
        "body": result.joke_body,
        "confidence": {
            "body": round(result.confidence_body, 4),
            "title": round(result.confidence_title, 4),
        },
        "needs_review": result.needs_review,
    }


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def process(eml_path: str, logger: logging.Logger) -> dict:
    """Run the full extraction pipeline for one .eml file."""
    logger.info("Processing: %s", eml_path)

    # --- Stage 1 ---
    try:
        prep = preprocess(eml_path)
    except FileNotFoundError:
        msg = f"File not found: {eml_path}"
        logger.error(msg)
        return _failure(msg)
    except Exception:
        msg = "Stage 1 preprocessing failed"
        logger.error("%s\n%s", msg, traceback.format_exc())
        return _failure(msg)

    logger.info(
        "Stage 1 complete | from=%s | subject=%r | segments=%d | total_chars=%d",
        prep.from_addr,
        prep.subject,
        len(prep.segments),
        sum(len(s) for s in prep.segments),
    )
    for i, seg in enumerate(prep.segments, 1):
        logger.debug("Stage 1 segment %d/%d:\n%s", i, len(prep.segments), seg)

    if not prep.segments:
        msg = "No text content found in email after preprocessing"
        logger.warning(msg)
        return _failure(msg)

    # --- Stage 2 ---
    try:
        result = extract_joke(prep.segments, prep.subject)
    except Exception:
        msg = "Stage 2 LLM extraction failed"
        logger.error("%s\n%s", msg, traceback.format_exc())
        return _failure(msg)

    if result.no_joke_found:
        msg = "LLM found no joke in email content"
        logger.info(
            "No joke found | from=%s | subject=%r", prep.from_addr, prep.subject
        )
        return _failure(msg)

    output = _success(prep, result)

    logger.info(
        "Extracted | from=%s | title=%r | title_source=%s | "
        "body_conf=%.2f | title_conf=%.2f | needs_review=%s | body_preview=%r",
        prep.from_addr,
        result.title,
        result.title_source,
        result.confidence_body,
        result.confidence_title,
        result.needs_review,
        result.joke_body[:200],
    )

    return output


def main() -> None:
    logger = _setup_logging()

    if len(sys.argv) != 2:
        _emit(_failure("Usage: python extract.py <path_to_email.eml>"))
        sys.exit(0)

    eml_path = sys.argv[1]
    output = process(eml_path, logger)
    _emit(output)
    sys.exit(0)


if __name__ == "__main__":
    main()
