"""
update_fixtures.py — Regenerate stage1.json and stage1.txt for all fixture cases.

Run this whenever preprocess.py changes or new .eml files are added:

    python tests/update_fixtures.py

For each tests/fixtures/<case>/email.eml:
  1. Runs preprocess() on it
  2. Finds a collision-free separator string
  3. Writes stage1.json
  4. Writes stage1.txt
"""

import json
import os
import sys
from pathlib import Path

# Allow importing from project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from preprocess import preprocess
from config import SEP_MARKER


FIXTURES_DIR = Path(__file__).parent / "fixtures"


def _find_separator(segments: list[str]) -> str:
    """Return a separator string that does not appear in any segment."""
    candidate = SEP_MARKER
    n = 1
    while any(candidate in seg for seg in segments):
        candidate = f"<<<SEP_{n}>>>"
        n += 1
    return candidate


def _write_stage1_json(dest: Path, segments: list[str], subject: str) -> None:
    data = {
        "segments": segments,
        "subject": subject,
    }
    dest.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def _write_stage1_txt(
    dest: Path,
    segments: list[str],
    subject: str,
    separator: str,
) -> None:
    lines = []
    lines.append(f"Subject: {subject}")
    lines.append(f"Segments: {len(segments)}")
    if len(segments) > 1:
        lines.append(f"Content-Separator: {separator}")
    lines.append("")  # blank line before body

    body = f"\n{separator}\n".join(segments)
    lines.append(body)

    dest.write_text("\n".join(lines), encoding="utf-8")


def process_case(case_dir: Path) -> str:
    """Process a single fixture case. Returns a status string."""
    eml_path = case_dir / "email.eml"
    if not eml_path.exists():
        return "SKIP  (no email.eml)"

    try:
        result = preprocess(str(eml_path))
    except Exception as exc:
        return f"ERROR ({exc})"

    separator = _find_separator(result.segments)

    _write_stage1_json(
        case_dir / "stage1.json",
        result.segments,
        result.subject,
    )
    _write_stage1_txt(
        case_dir / "stage1.txt",
        result.segments,
        result.subject,
        separator,
    )

    seg_word = "segment" if len(result.segments) == 1 else "segments"
    sep_note = f" (separator: {separator})" if separator != SEP_MARKER else ""
    return f"OK    ({len(result.segments)} {seg_word}{sep_note})"


def main() -> None:
    if not FIXTURES_DIR.exists():
        print(f"fixtures dir not found: {FIXTURES_DIR}")
        sys.exit(1)

    cases = sorted(p for p in FIXTURES_DIR.iterdir() if p.is_dir())
    if not cases:
        print("No fixture directories found.")
        return

    width = max(len(c.name) for c in cases)
    for case_dir in cases:
        status = process_case(case_dir)
        print(f"  {case_dir.name:<{width}}  {status}")


if __name__ == "__main__":
    main()
