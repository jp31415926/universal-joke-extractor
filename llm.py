"""Stage 2: LLM-based joke extraction via Ollama."""

import json
import logging
import re
from typing import Any

import ollama

_logger = logging.getLogger("joke_extractor")

from config import (
    CONFIDENCE_THRESHOLD_BODY,
    CONFIDENCE_THRESHOLD_TITLE,
    MIN_BODY_LENGTH,
    OLLAMA_CTX_MIN,
    OLLAMA_HOST,
    OLLAMA_KEEP_ALIVE,
    OLLAMA_MODEL,
    OLLAMA_MODEL_CTX,
    OLLAMA_TIMEOUT,
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
            "description": "The best title for this joke. Must be a short phrase (under 80 characters). Never put joke content here.",
            "maxLength": 80,
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
You are a joke extraction assistant. You receive cleaned email content submitted to a joke newsletter. Extract the joke text, choose a title, and report confidence.

━━ WHAT TO EXTRACT ━━

Extract the joke completely — from its very first line to its very last line. Include the setup, all items, and the punchline. Never truncate.

LIST JOKES: If the joke is a list (numbered items, multiple one-liners sharing a theme, multiple vignettes by different contributors, etc.), extract EVERY item from first to last. Never pick just one item or only the final punchline.

ATTRIBUTED QUOTES: If the joke presents quotes from multiple people (e.g., children's quotes each followed by a name/age line like "Gregory, age 5" or "-Olive, age 9"), extract each quote AND its attribution line — they are part of the joke.

━━ WHAT TO OMIT ━━

Strip these — they are NOT part of the joke:
- Submitter notes addressed to the newsletter editors: e.g., "I thought you'd enjoy this", "Seeing that you used my last submission, here's another", "Feel free to edit."
- Submitter initials, name sign-offs, or closing signatures at the end (e.g., "BCS", "- Barry", "-- John Smith").
- Forwarding boilerplate ("FW:", "check this out", "thought you'd enjoy this").

Do NOT strip closing sentences that are part of the joke's own voice, even if they reference the joke itself.

━━ MULTI-SEGMENT EMAILS ━━

The email may contain multiple segments separated by "{SEP_MARKER}". The joke is in the most substantial segment — ignore short ad or navigation fragments.

━━ TITLE SOURCE RULES ━━

"body_internal" — A title or heading is LITERALLY present as a heading line in the joke text itself (e.g., "CONFUCIUS DID NOT SAY...", "To Women Everywhere From Men Who Have Had Enough!"). Copy it verbatim. Never use body_internal for a title you invented yourself.

"subject" — No title heading appears in the joke body. Use the email Subject line as-is. Prefer this over generating a new title, even for informal subjects.

"generated" — Use only if the subject is completely irrelevant to the joke AND no title appears in the body. You are inventing the title yourself.

━━ OTHER RULES ━━

- title must be short (under 80 characters, typically 2-8 words). Never put joke content in title.
- Do not fix typos or improve the joke text. Extract verbatim.
- Confidence: 1.0 = certain, 0.5 = unsure, below 0.4 = guessing.
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


def _estimate_ctx(system_prompt: str, user_message: str) -> int:
    """
    Calculate num_ctx for this call, scaled to actual content size.

    Token estimation: 4 chars ≈ 1 token (empirically confirmed to within ~6%
    across test cases using actual prompt_eval_count from Ollama responses).

    The 3× multiplier reserves 2× the input size as output headroom — enough
    for joke bodies that approach the length of the full email content.
    Without sufficient headroom, llama models may stop generating early even
    when the content technically fits, due to rope-scaling attention effects.

    Result is rounded up to the nearest 1024 and clamped to
    [OLLAMA_CTX_MIN, OLLAMA_MODEL_CTX].
    """
    input_tokens = (len(system_prompt) + len(user_message)) // 4
    needed = input_tokens * 3
    rounded = ((needed + 1023) // 1024) * 1024
    return max(OLLAMA_CTX_MIN, min(rounded, OLLAMA_MODEL_CTX))


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
    client = ollama.Client(host=OLLAMA_HOST, timeout=OLLAMA_TIMEOUT)
    user_message = _build_user_message(segments, subject)

    _logger.debug("LLM user prompt:\n%s", user_message)

    response = client.chat(
        model=OLLAMA_MODEL,
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
        format=_EXTRACTION_SCHEMA,
        options={"temperature": 0.1, "num_ctx": _estimate_ctx(_SYSTEM_PROMPT, user_message)},
        keep_alive=OLLAMA_KEEP_ALIVE,
    )

    prompt_tokens = getattr(response, "prompt_eval_count", None)
    output_tokens = getattr(response, "eval_count", None)
    estimated_input = (len(_SYSTEM_PROMPT) + len(user_message)) // 4
    _logger.debug(
        "Token usage — estimated_input: %s  actual_prompt: %s  actual_output: %s  "
        "ratio(actual/estimated): %.2f  num_ctx: %s",
        estimated_input,
        prompt_tokens,
        output_tokens,
        (prompt_tokens / estimated_input) if prompt_tokens and estimated_input else 0,
        _estimate_ctx(_SYSTEM_PROMPT, user_message),
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
