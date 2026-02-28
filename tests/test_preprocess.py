"""
test_preprocess.py — Unit tests for preprocess.py (Stage 1).

Covers deterministic text-processing logic using synthetic string inputs.
No .eml files or Ollama instance required.

Run with:
    pytest tests/test_preprocess.py
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from preprocess import (
    _strip_signature,
    _strip_quotes,
    _replace_separators,
    _strip_forwarding_headers,
    _clean_segment,
    _html_to_text,
    _bytes_to_str,
)
from config import SEP_MARKER


# ---------------------------------------------------------------------------
# Signature stripping
# ---------------------------------------------------------------------------

class TestStripSignature:
    def test_standard_sig_delimiter_removed(self):
        text = "Here is a joke.\n\n-- \nJohn Smith\njohn@example.com"
        result = _strip_signature(text)
        assert "John Smith" not in result
        assert "Here is a joke." in result

    def test_sig_delimiter_bare_also_treated_as_sig(self):
        # "--" without trailing space is also matched by ^--\s*$ and treated as sig delimiter
        text = "Some text.\n\n--\nSig content here"
        result = _strip_signature(text)
        assert "Sig content here" not in result
        assert "Some text." in result

    def test_footer_phrase_sent_from_my_stripped(self):
        text = "Why did the chicken cross the road?\nTo get to the other side.\nSent from my iPhone"
        result = _strip_signature(text)
        assert "Sent from my iPhone" not in result
        assert "To get to the other side." in result

    def test_footer_phrase_get_outlook_stripped(self):
        text = "A joke.\nGet Outlook for iOS"
        result = _strip_signature(text)
        assert "Get Outlook" not in result
        assert "A joke." in result

    def test_footer_phrase_unsubscribe_stripped(self):
        text = "Funny content here.\nTo unsubscribe from this list, click here."
        result = _strip_signature(text)
        assert "unsubscribe" not in result.lower()
        assert "Funny content here." in result

    def test_no_signature_unchanged(self):
        text = "Why don't scientists trust atoms?\nBecause they make up everything!"
        result = _strip_signature(text)
        assert result == text

    def test_multiple_footer_lines_all_removed(self):
        text = "The joke body.\n\n-- \nJane Doe\njane@example.com\nPhone: 555-1234"
        result = _strip_signature(text)
        assert "Jane Doe" not in result
        assert "The joke body." in result

    def test_empty_string(self):
        assert _strip_signature("") == ""


# ---------------------------------------------------------------------------
# Quote stripping
# ---------------------------------------------------------------------------

class TestStripQuotes:
    def test_single_level_quotes_removed(self):
        text = "> This was forwarded\n> to you by a friend"
        result = _strip_quotes(text)
        assert ">" not in result
        assert "This was forwarded" in result

    def test_double_level_quotes_removed(self):
        text = ">> Deeply quoted text"
        result = _strip_quotes(text)
        assert ">" not in result
        assert "Deeply quoted text" in result

    def test_triple_level_quotes_removed(self):
        text = ">>> Very deep quote"
        result = _strip_quotes(text)
        assert ">" not in result
        assert "Very deep quote" in result

    def test_spaced_multilevel_quotes_removed(self):
        # "> > text" — levels separated by spaces (common in forwarded email)
        text = "> > Deeply forwarded content"
        result = _strip_quotes(text)
        assert ">" not in result
        assert "Deeply forwarded content" in result

    def test_leading_space_before_quote_removed(self):
        # " >text" — line starts with whitespace then ">" (seen in real emails)
        text = " >Indented quote"
        result = _strip_quotes(text)
        assert ">" not in result
        assert "Indented quote" in result

    def test_non_quoted_lines_unchanged(self):
        text = "Normal line\n> Quoted line\nAnother normal"
        result = _strip_quotes(text)
        assert "Normal line" in result
        assert "Another normal" in result
        assert ">" not in result

    def test_empty_string(self):
        assert _strip_quotes("") == ""

    def test_quote_with_no_space_removed(self):
        text = ">No space after marker"
        result = _strip_quotes(text)
        assert ">" not in result
        assert "No space after marker" in result


# ---------------------------------------------------------------------------
# Separator replacement
# ---------------------------------------------------------------------------

class TestReplaceSeparators:
    def test_equals_line_replaced(self):
        text = "Part one\n====\nPart two"
        result = _replace_separators(text)
        assert SEP_MARKER in result
        assert "====" not in result

    def test_dashes_replaced(self):
        text = "Part one\n----\nPart two"
        result = _replace_separators(text)
        assert SEP_MARKER in result
        assert "----" not in result

    def test_stars_replaced(self):
        text = "Part one\n****\nPart two"
        result = _replace_separators(text)
        assert SEP_MARKER in result

    def test_underscores_replaced(self):
        text = "Part one\n____\nPart two"
        result = _replace_separators(text)
        assert SEP_MARKER in result

    def test_short_separator_not_replaced(self):
        # Fewer than 4 chars — not treated as a separator
        text = "Part one\n===\nPart two"
        result = _replace_separators(text)
        assert SEP_MARKER not in result

    def test_long_separator_replaced(self):
        text = "Part one\n" + "=" * 40 + "\nPart two"
        result = _replace_separators(text)
        assert SEP_MARKER in result

    def test_no_separator_unchanged(self):
        text = "Just some text with no separator lines."
        result = _replace_separators(text)
        assert result == text


# ---------------------------------------------------------------------------
# Forwarding header stripping
# ---------------------------------------------------------------------------

class TestStripForwardingHeaders:
    def test_from_header_stripped(self):
        text = "Original content\nFrom: someone@example.com\nMore content"
        result = _strip_forwarding_headers(text)
        assert "someone@example.com" not in result
        assert "Original content" in result

    def test_subject_header_stripped(self):
        text = "-----\nSubject: Re: Funny joke\n-----"
        result = _strip_forwarding_headers(text)
        assert "Subject: Re: Funny joke" not in result

    def test_date_header_stripped(self):
        text = "Body text\nDate: Mon, 1 Jan 2024 12:00:00 +0000"
        result = _strip_forwarding_headers(text)
        assert "Date:" not in result
        assert "Body text" in result

    def test_sent_header_stripped(self):
        text = "Content\nSent: Monday, January 1, 2024"
        result = _strip_forwarding_headers(text)
        assert "Sent:" not in result

    def test_to_header_stripped(self):
        text = "Message\nTo: recipient@example.com"
        result = _strip_forwarding_headers(text)
        assert "recipient@example.com" not in result

    def test_case_insensitive(self):
        text = "Text\nFROM: UPPER@EXAMPLE.COM"
        result = _strip_forwarding_headers(text)
        assert "UPPER@EXAMPLE.COM" not in result

    def test_regular_text_from_not_stripped(self):
        # "From" not in a header context (not at line start with colon)
        text = "I heard this joke from my friend."
        result = _strip_forwarding_headers(text)
        assert "I heard this joke from my friend." in result


# ---------------------------------------------------------------------------
# HTML fallback path
# ---------------------------------------------------------------------------

class TestHtmlToText:
    def test_basic_paragraph(self):
        html = "<p>Hello world</p>"
        result = _html_to_text(html)
        assert "Hello world" in result

    def test_links_ignored(self):
        html = '<p>Click <a href="http://example.com">here</a> to subscribe</p>'
        result = _html_to_text(html)
        assert "http://example.com" not in result
        assert "here" in result

    def test_images_ignored(self):
        html = '<p>Text</p><img src="image.png" alt="funny pic">'
        result = _html_to_text(html)
        assert "image.png" not in result
        assert "funny pic" not in result
        assert "Text" in result

    def test_line_breaks_preserved(self):
        html = "<p>Line one</p><p>Line two</p>"
        result = _html_to_text(html)
        assert "Line one" in result
        assert "Line two" in result

    def test_html_entities_decoded(self):
        html = "<p>Q: Why &amp; How?</p>"
        result = _html_to_text(html)
        assert "&" in result

    def test_bold_text_content_preserved(self):
        html = "<p>This is <b>important</b>.</p>"
        result = _html_to_text(html)
        assert "important" in result


# ---------------------------------------------------------------------------
# Clean segment integration
# ---------------------------------------------------------------------------

class TestCleanSegment:
    def test_full_pipeline_removes_sig_and_quotes(self):
        raw = (
            b"> Forwarded joke:\n"
            b"Why did the scarecrow win an award?\n"
            b"Because he was outstanding in his field!\n"
            b"\n-- \n"
            b"Sender Name\n"
        )
        result = _clean_segment(raw, "utf-8")
        assert "Sender Name" not in result
        assert ">" not in result
        assert "outstanding in his field" in result

    def test_separator_replaced_in_pipeline(self):
        raw = b"Part A\n======\nPart B"
        result = _clean_segment(raw, "utf-8")
        assert SEP_MARKER in result
        assert "======" not in result

    def test_iso8859_charset(self):
        # iso-8859-1 encoding of "café"
        raw = "caf\xe9".encode("iso-8859-1")
        result = _clean_segment(raw, "iso-8859-1")
        assert "caf" in result  # character decoded successfully

    def test_empty_bytes_returns_empty(self):
        result = _clean_segment(b"", "utf-8")
        assert result == ""

    def test_whitespace_only_returns_empty(self):
        result = _clean_segment(b"   \n\n\t  ", "utf-8")
        assert result == ""
