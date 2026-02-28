"""
import_emls.py — Bulk-import .eml files into the fixture tree.

Usage:
    python tests/import_emls.py /path/to/eml/folder

For each *.eml in the given directory:
  1. Derives a fixture case name by slugifying the filename (lowercase,
     non-alphanumeric → underscore, collapsed, max 20 chars).
  2. Skips if tests/fixtures/<slug>/ already exists.
  3. Creates the fixture dir, copies the file to email.eml.
  4. Redacts all email addresses in the copied email.eml (headers + body)
     by replacing them with REDACTED@GCFL.net.
  5. Runs preprocess() and writes stage1.json + stage1.txt.
  6. Writes a template expected.txt for manual completion.
"""

import json
import re
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from preprocess import preprocess
from config import SEP_MARKER

FIXTURES_DIR = Path(__file__).parent / "fixtures"
MAX_SLUG_LEN = 20


def slugify(name: str) -> str:
    """Convert a filename stem to a safe fixture directory name."""
    slug = name.lower()
    slug = re.sub(r"[^a-z0-9]+", "_", slug)  # non-alphanumeric → underscore
    slug = slug.strip("_")
    return slug[:MAX_SLUG_LEN].rstrip("_")


# Headers to strip entirely (including folded continuation lines).
# Patterns are matched case-insensitively against the header name only.
_STRIP_HEADER_RE = re.compile(
    r"^(?:"
    r"Received"
    r"|X-[^:]*"
    r"|ARC-[^:]*"
    r"|Authentication-Results"
    r"|Received-SPF"
    r"|DKIM-Signature"
    r"|Resent-[^:]*"
    r"):",
    re.IGNORECASE,
)


def _strip_headers(eml_path: Path) -> None:
    """Remove privacy-sensitive headers (including their folded continuations) in-place."""
    text = eml_path.read_bytes().decode("iso-8859-1")

    # Split into header block and body — blank line is the separator.
    if "\r\n\r\n" in text:
        header_block, sep, body = text.partition("\r\n\r\n")
        line_sep = "\r\n"
    elif "\n\n" in text:
        header_block, sep, body = text.partition("\n\n")
        line_sep = "\n"
    else:
        return  # malformed — leave untouched

    lines = header_block.split(line_sep)
    kept: list[str] = []
    skip = False
    for line in lines:
        if line and line[0] in (" ", "\t"):
            # Folded continuation — inherit the skip decision of its parent.
            if not skip:
                kept.append(line)
        else:
            skip = bool(_STRIP_HEADER_RE.match(line))
            if not skip:
                kept.append(line)

    eml_path.write_bytes((line_sep.join(kept) + sep + body).encode("iso-8859-1"))


_EMAIL_RE = re.compile(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}")
_REDACTED = "REDACTED@GCFL.net"

_IPV4_RE = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")
_IPV4_REDACTED = "x.x.x.x"

_IPV6_RE = re.compile(
    r"(?:[0-9a-fA-F]{1,4}:){7}[0-9a-fA-F]{1,4}"                    # 1:2:3:4:5:6:7:8
    r"|(?:[0-9a-fA-F]{1,4}:){1,7}:"                                  # 1::  through  1:2:3:4:5:6:7::
    r"|(?:[0-9a-fA-F]{1,4}:){1,6}:[0-9a-fA-F]{1,4}"                 # 1::8  through  1:2:3:4:5:6::8
    r"|(?:[0-9a-fA-F]{1,4}:){1,5}(?::[0-9a-fA-F]{1,4}){1,2}"       # 1::7:8  etc.
    r"|(?:[0-9a-fA-F]{1,4}:){1,4}(?::[0-9a-fA-F]{1,4}){1,3}"
    r"|(?:[0-9a-fA-F]{1,4}:){1,3}(?::[0-9a-fA-F]{1,4}){1,4}"
    r"|(?:[0-9a-fA-F]{1,4}:){1,2}(?::[0-9a-fA-F]{1,4}){1,5}"
    r"|[0-9a-fA-F]{1,4}:(?::[0-9a-fA-F]{1,4}){1,6}"
    r"|:(?::[0-9a-fA-F]{1,4}){1,7}"                                  # ::2  through  ::2:3:4:5:6:7:8
    r"|::",                                                            # ::
)
_IPV6_REDACTED = "x:x::x"


def _redact_emails(eml_path: Path) -> int:
    """Replace email addresses and IP addresses in eml_path in-place.
    Returns the total number of replacements made."""
    raw = eml_path.read_bytes()
    text = raw.decode("iso-8859-1")
    text, n_email = _EMAIL_RE.subn(_REDACTED, text)
    text, n_ipv4 = _IPV4_RE.subn(_IPV4_REDACTED, text)
    text, n_ipv6 = _IPV6_RE.subn(_IPV6_REDACTED, text)
    count = n_email + n_ipv4 + n_ipv6
    if count:
        eml_path.write_bytes(text.encode("iso-8859-1"))
    return count


def _find_separator(segments: list[str]) -> str:
    candidate = SEP_MARKER
    n = 1
    while any(candidate in seg for seg in segments):
        candidate = f"<<<SEP_{n}>>>"
        n += 1
    return candidate


def _write_stage1_json(dest: Path, segments: list[str], subject: str) -> None:
    dest.write_text(
        json.dumps({"segments": segments, "subject": subject},
                   indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def _write_stage1_txt(
    dest: Path, segments: list[str], subject: str, separator: str
) -> None:
    lines = [
        f"Subject: {subject}",
        f"Segments: {len(segments)}",
    ]
    if len(segments) > 1:
        lines.append(f"Content-Separator: {separator}")
    lines.append("")
    lines.append(f"\n{separator}\n".join(segments))
    dest.write_text("\n".join(lines), encoding="utf-8")


def _write_expected_txt(dest: Path, subject: str) -> None:
    lines = [
        f"Title: {subject}",
        "Title-Source: body_internal|subject|generated",
        "No-Joke-Found: true|false",
        "",
        "<joke text>",
    ]
    dest.write_text("\n".join(lines), encoding="utf-8")


def import_eml(eml_path: Path) -> str:
    """Import one .eml file. Returns a one-line status string."""
    slug = slugify(eml_path.stem)
    if not slug:
        return f"SKIP  {eml_path.name!r} — could not derive a valid slug"

    case_dir = FIXTURES_DIR / slug
    if case_dir.exists():
        return f"SKIP  {eml_path.name!r} → {slug}/ already exists"

    case_dir.mkdir(parents=True)
    dest_eml = case_dir / "email.eml"
    shutil.copy2(eml_path, dest_eml)
    _strip_headers(dest_eml)
    _redact_emails(dest_eml)

    try:
        result = preprocess(str(dest_eml))
    except Exception as exc:
        # Leave the dir so the user can inspect; don't write stage1 files
        return f"ERROR {eml_path.name!r} → {slug}/ — preprocess failed: {exc}"

    separator = _find_separator(result.segments)
    _write_stage1_json(case_dir / "stage1.json", result.segments, result.subject)
    _write_stage1_txt(case_dir / "stage1.txt", result.segments, result.subject, separator)
    _write_expected_txt(case_dir / "expected.txt", result.subject)

    seg_word = "segment" if len(result.segments) == 1 else "segments"
    return f"OK    {eml_path.name!r} → {slug}/ ({len(result.segments)} {seg_word})"


def main() -> None:
    if len(sys.argv) != 2:
        print(f"Usage: python {Path(__file__).name} <directory>")
        sys.exit(1)

    source_dir = Path(sys.argv[1])
    if not source_dir.is_dir():
        print(f"Error: not a directory: {source_dir}")
        sys.exit(1)

    eml_files = sorted(source_dir.glob("*.eml"))
    if not eml_files:
        print(f"No .eml files found in {source_dir}")
        sys.exit(0)

    FIXTURES_DIR.mkdir(parents=True, exist_ok=True)

    ok = skip = error = 0
    for eml_path in eml_files:
        status = import_eml(eml_path)
        print(status)
        if status.startswith("OK"):
            ok += 1
        elif status.startswith("SKIP"):
            skip += 1
        else:
            error += 1

    print()
    parts = [f"{ok} imported"]
    if skip:
        parts.append(f"{skip} skipped")
    if error:
        parts.append(f"{error} errors")
    print(", ".join(parts))


if __name__ == "__main__":
    main()
