"""Stage 2: LLM-based joke extraction via Ollama."""

import json
import logging
from typing import Any

import ollama

_logger = logging.getLogger("joke_extractor")

from config import (
    CONFIDENCE_THRESHOLD_BODY,
    CONFIDENCE_THRESHOLD_TITLE,
    MIN_BODY_LENGTH,
    OLLAMA_HOST,
    OLLAMA_MODEL,
    SEP_MARKER,
)


# ---------------------------------------------------------------------------
# JSON schema for structured Ollama output
# ---------------------------------------------------------------------------

_EXTRACTION_SCHEMA = {
    "type": "object",
    "properties": {
        "joke_body": {
            "type": "string",
            "description": "The full extracted joke text, verbatim from the email.",
        },
        "title": {
            "type": "string",
            "description": "The best title for this joke.",
        },
        "title_source": {
            "type": "string",
            "enum": ["body_internal", "subject", "generated"],
            "description": "Where the title came from.",
        },
        "confidence": {
            "type": "object",
            "properties": {
                "body": {
                    "type": "number",
                    "description": "Confidence 0.0-1.0 that joke_body is correctly extracted.",
                },
                "title": {
                    "type": "number",
                    "description": "Confidence 0.0-1.0 that title is the best choice.",
                },
            },
            "required": ["body", "title"],
        },
        "no_joke_found": {
            "type": "boolean",
            "description": "True if no joke could be identified in the email.",
        },
    },
    "required": ["joke_body", "title", "title_source", "confidence", "no_joke_found"],
}

_SYSTEM_PROMPT = f"""\
You are a joke extraction assistant. You receive cleaned email content that was submitted to a joke newsletter. Your job is to:

1. Find and extract the joke body — the actual joke text, verbatim, without surrounding commentary, signatures, or meta-text.
2. Choose the best title for the joke. Consider these sources in order of preference:
   - A title explicitly stated inside the joke body (e.g. "Subject: The Talking Dog")
   - The email Subject line, if it clearly describes the joke
   - Generate a short, descriptive title yourself
3. Report your confidence (0.0 to 1.0) for both the extracted body and the chosen title.
4. If the email contains no joke (it's spam, a test, a request, an error, or unrelated content), set no_joke_found to true.

Guidelines:
- One-liners and short jokes are valid — extract them fully.
- If a joke is embedded in commentary ("Here's a funny one I heard..."), extract only the joke, not the intro.
- The email may contain multiple text segments separated by "{SEP_MARKER}". The joke is likely in one of these segments — identify the best one.
- Do not add, edit, or improve the joke text. Extract it as-is.
- Confidence of 1.0 means you are certain. 0.5 means you are unsure. Below 0.4 means you are guessing.
"""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

class ExtractionResult:
    """Result of LLM joke extraction."""

    def __init__(
        self,
        joke_body: str,
        title: str,
        title_source: str,
        confidence_body: float,
        confidence_title: float,
        no_joke_found: bool,
    ):
        self.joke_body = joke_body
        self.title = title
        self.title_source = title_source
        self.confidence_body = confidence_body
        self.confidence_title = confidence_title
        self.no_joke_found = no_joke_found

    @property
    def needs_review(self) -> bool:
        if len(self.joke_body) < MIN_BODY_LENGTH:
            return True
        return (
            self.confidence_body < CONFIDENCE_THRESHOLD_BODY
            or self.confidence_title < CONFIDENCE_THRESHOLD_TITLE
        )


def _build_user_message(segments: list[str], subject: str) -> str:
    """Format cleaned segments and subject into the LLM user message."""
    lines = [f"Subject line: {subject}" if subject else "Subject line: (none)", ""]

    if not segments:
        lines.append("(No text content found in email)")
    elif len(segments) == 1:
        lines.append("Email body:")
        lines.append(segments[0])
    else:
        lines.append(f"Email contains {len(segments)} text segment(s):")
        for i, seg in enumerate(segments, 1):
            lines.append(f"\n--- Segment {i} ---")
            lines.append(seg)

    return "\n".join(lines)


def extract_joke(segments: list[str], subject: str) -> ExtractionResult:
    """
    Call Ollama to extract the joke from preprocessed email segments.

    Args:
        segments: List of cleaned text segments from Stage 1.
        subject: Original email subject line.

    Returns:
        ExtractionResult with joke data and confidence scores.

    Raises:
        ollama.ResponseError: On Ollama API errors.
        ValueError: If the LLM response cannot be parsed.
    """
    client = ollama.Client(host=OLLAMA_HOST)
    user_message = _build_user_message(segments, subject)

    _logger.debug("LLM user prompt:\n%s", user_message)

    response = client.chat(
        model=OLLAMA_MODEL,
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
        format=_EXTRACTION_SCHEMA,
        options={"temperature": 0.1},  # low temp for deterministic extraction
    )

    raw_content = response.message.content
    _logger.debug("LLM raw response:\n%s", raw_content)
    try:
        data: dict[str, Any] = json.loads(raw_content)
    except json.JSONDecodeError as exc:
        raise ValueError(f"LLM returned non-JSON response: {raw_content!r}") from exc

    confidence = data.get("confidence", {})

    return ExtractionResult(
        joke_body=data.get("joke_body", ""),
        title=data.get("title", ""),
        title_source=data.get("title_source", "generated"),
        confidence_body=float(confidence.get("body", 0.0)),
        confidence_title=float(confidence.get("title", 0.0)),
        no_joke_found=bool(data.get("no_joke_found", False)),
    )
