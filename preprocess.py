"""Stage 1: Deterministic preprocessing of raw .eml files."""

import email
import email.utils
import email.policy
import re
import quopri
from email import message_from_bytes, message_from_string
from email.message import Message
from typing import NamedTuple

import html2text

from config import SEP_MARKER


# ---------------------------------------------------------------------------
# Unicode normalization maps
# ---------------------------------------------------------------------------

_UNICODE_REPLACEMENTS = {
    "\u2018": "'",   # left single quotation mark
    "\u2019": "'",   # right single quotation mark
    "\u201c": '"',   # left double quotation mark
    "\u201d": '"',   # right double quotation mark
    "\u2013": "-",   # en dash
    "\u2014": "--",  # em dash
    "\u00ad": "",    # soft hyphen (invisible)
    "\u00a0": " ",   # non-breaking space
    "\u2026": "...", # ellipsis
}

_UNICODE_TRANS = str.maketrans(_UNICODE_REPLACEMENTS)

# Separator patterns: lines that are only repeated punctuation (===, ---, ***)
_SEP_RE = re.compile(r"^([=\-*_#~]{4,})\s*$", re.MULTILINE)

# Leading quote markers: handles "> > >", " >", ">> ", leading spaces, etc.
_QUOTE_RE = re.compile(r"^\s*(>\s*)+", re.MULTILINE)

# Email signature delimiter
_SIG_DELIM_RE = re.compile(r"^--\s*$", re.MULTILINE)

# Common footer/signature phrases (case-insensitive)
_FOOTER_PHRASES = re.compile(
    r"(sent from my |get outlook|unsubscribe|this email was sent"
    r"|this message was sent|to unsubscribe|click here to unsubscribe"
    r"|powered by|confidentiality notice|disclaimer)",
    re.IGNORECASE,
)

# Forwarding headers that appear mid-body (e.g. "-----Original Message-----")
_FWD_HEADER_RE = re.compile(
    r"^[ \t]*(from|to|date|sent|subject)\s*:\s*.+$",
    re.MULTILINE | re.IGNORECASE,
)

# All-caps lines (likely footers/banners), 4+ words, no lowercase
_ALL_CAPS_LINE_RE = re.compile(r"^(?:[A-Z0-9 \t,.'!?&-]{20,})$", re.MULTILINE)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

class PreprocessResult(NamedTuple):
    segments: list[str]   # cleaned text segments, one per MIME part
    subject: str          # original Subject header value
    from_addr: str        # parsed submitter email address


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _decode_payload(part: Message) -> bytes:
    """Return raw decoded bytes for a MIME part, handling quoted-printable."""
    charset = part.get_content_charset() or "iso-8859-1"
    payload = part.get_payload(decode=True)  # handles base64 + QP automatically
    return payload or b""


def _bytes_to_str(raw: bytes, charset: str = "iso-8859-1") -> str:
    """Decode bytes to str using the given charset, replacing unmappable chars."""
    try:
        return raw.decode(charset, errors="replace")
    except (LookupError, UnicodeDecodeError):
        return raw.decode("iso-8859-1", errors="replace")


def _html_to_text(html: str) -> str:
    """Convert HTML to plain text, preserving paragraph structure."""
    converter = html2text.HTML2Text()
    converter.ignore_links = True
    converter.ignore_images = True
    converter.ignore_emphasis = False
    converter.body_width = 0  # don't wrap lines
    return converter.handle(html)


def _normalize_unicode(text: str) -> str:
    """Replace common unicode punctuation with ASCII equivalents."""
    return text.translate(_UNICODE_TRANS)


def _normalize_whitespace(text: str) -> str:
    """Normalize line endings and collapse excessive blank lines."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    # Collapse 3+ consecutive blank lines to 2
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text


def _replace_separators(text: str) -> str:
    """Replace separator lines with the SEP_MARKER constant."""
    return _SEP_RE.sub(SEP_MARKER, text)


def _strip_quotes(text: str) -> str:
    """Remove leading > quote markers from each line."""
    return _QUOTE_RE.sub("", text)


def _strip_forwarding_headers(text: str) -> str:
    """Remove inline forwarding headers (From:, To:, Date:, Subject:) mid-body."""
    return _FWD_HEADER_RE.sub("", text)


def _strip_signature(text: str) -> str:
    """
    Remove email signature blocks.

    Strategy:
    1. If the standard `-- ` delimiter appears, cut everything after it.
    2. Walk lines from the bottom; drop lines matching footer phrases or
       all-caps banner patterns until we hit real content.
    """
    # Standard sig delimiter
    match = _SIG_DELIM_RE.search(text)
    if match:
        text = text[: match.start()]

    # Bottom-up footer stripping
    lines = text.splitlines()
    cutoff = len(lines)
    for i in range(len(lines) - 1, -1, -1):
        line = lines[i].strip()
        if not line:
            continue
        if _FOOTER_PHRASES.search(line) or _ALL_CAPS_LINE_RE.match(line):
            cutoff = i
        else:
            break

    return "\n".join(lines[:cutoff])


def _clean_segment(raw_bytes: bytes, charset: str) -> str:
    """Apply the full normalization pipeline to one text segment."""
    text = _bytes_to_str(raw_bytes, charset)
    text = _normalize_unicode(text)
    text = _normalize_whitespace(text)
    text = _strip_quotes(text)
    text = _strip_forwarding_headers(text)
    text = _replace_separators(text)
    text = _strip_signature(text)
    text = _normalize_whitespace(text)  # re-run after removal passes
    return text.strip()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def collect_text_parts(msg: Message) -> list[tuple[bytes, str]]:
    """
    Walk the MIME tree and collect all text/plain parts.
    Falls back to text/html (converted) if no plain parts exist.

    Returns list of (raw_bytes, charset) tuples.
    """
    plain_parts: list[tuple[bytes, str]] = []
    html_parts: list[tuple[bytes, str]] = []

    for part in msg.walk():
        ctype = part.get_content_type()
        if part.get_content_maintype() == "multipart":
            continue
        charset = part.get_content_charset() or "iso-8859-1"
        payload = part.get_payload(decode=True)
        if payload is None:
            continue

        if ctype == "text/plain":
            plain_parts.append((payload, charset))
        elif ctype == "text/html":
            html_parts.append((payload, charset))

    if plain_parts:
        return plain_parts

    # Fall back: convert HTML parts to plain text bytes
    result = []
    for raw, charset in html_parts:
        html_str = _bytes_to_str(raw, charset)
        plain = _html_to_text(html_str)
        result.append((plain.encode("iso-8859-1", errors="replace"), "iso-8859-1"))
    return result


def preprocess(eml_path: str) -> PreprocessResult:
    """
    Parse an .eml file and return cleaned text segments, subject, and sender.

    Args:
        eml_path: Filesystem path to the .eml file.

    Returns:
        PreprocessResult with segments, subject, from_addr.

    Raises:
        OSError: If the file cannot be read.
        email.errors.MessageError: If the file cannot be parsed.
    """
    with open(eml_path, "rb") as f:
        raw = f.read()

    msg = message_from_bytes(raw)

    # Extract headers
    subject = msg.get("Subject", "") or ""
    from_header = msg.get("From", "") or ""
    _, from_addr = email.utils.parseaddr(from_header)

    # Collect and clean all text parts
    parts = collect_text_parts(msg)
    segments = []
    for raw_bytes, charset in parts:
        cleaned = _clean_segment(raw_bytes, charset)
        if cleaned:
            segments.append(cleaned)

    return PreprocessResult(
        segments=segments,
        subject=subject.strip(),
        from_addr=from_addr.strip(),
    )
