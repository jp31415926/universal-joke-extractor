"""
run_eval.py — LLM extraction evaluation harness.

For each fixture dir containing both stage1.json and expected.txt:
  1. Loads stage1.json
  2. Calls extract_joke(segments, subject)
  3. Writes actual-stage2.txt (same format as expected.txt)
  4. Compares actual vs expected; prints PASS/FAIL table
  5. Exits with code 1 if any case fails

Usage:
    python tests/run_eval.py
    python tests/run_eval.py tests/fixtures/angels_explained  # single case
"""

import html
import json
import logging
import re
import sys
import difflib
from pathlib import Path

# Allow importing from project root
sys.path.insert(0, str(Path(__file__).parent.parent))

# Route joke_extractor debug logs to stderr so token usage is visible
logging.basicConfig(stream=sys.stderr, level=logging.DEBUG,
                    format="%(message)s")
logging.getLogger("joke_extractor").setLevel(logging.DEBUG)

from llm import extract_joke

FIXTURES_DIR = Path(__file__).parent / "fixtures"


# ---------------------------------------------------------------------------
# File format helpers
# ---------------------------------------------------------------------------

def _parse_result_file(path: Path) -> dict:
    """Parse expected.txt or actual-stage2.txt into a dict."""
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines()

    headers = {}
    body_lines = []
    in_body = False

    for line in lines:
        if in_body:
            body_lines.append(line)
        elif line == "":
            in_body = True
        elif ":" in line:
            key, _, val = line.partition(":")
            headers[key.strip()] = val.strip()
        else:
            # line with no colon — treat as start of body
            in_body = True
            body_lines.append(line)

    return {
        "title": headers.get("Title", ""),
        "title_source": headers.get("Title-Source", ""),
        "no_joke_found": headers.get("No-Joke-Found", "false").lower() == "true",
        "joke_body": "\n".join(body_lines).strip(),
    }


def _format_result_file(title: str, title_source: str, no_joke_found: bool, joke_body: str) -> str:
    """Render an actual-stage2.txt / expected.txt in canonical format."""
    lines = [
        f"Title: {title}",
        f"Title-Source: {title_source}",
        f"No-Joke-Found: {'true' if no_joke_found else 'false'}",
        "",
    ]
    if not no_joke_found:
        lines.append(joke_body)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Comparison logic
# ---------------------------------------------------------------------------

def _normalize_body(text: str) -> str:
    """Lowercase, decode HTML entities, strip punctuation, collapse whitespace → space-joined tokens."""
    text = html.unescape(text)
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    tokens = text.split()
    return " ".join(tokens)


def _compare(expected: dict, actual: dict) -> list[str]:
    """Return list of failure reasons (empty = PASS)."""
    failures = []

    if expected["no_joke_found"] != actual["no_joke_found"]:
        failures.append(
            f"No-Joke-Found mismatch: expected={expected['no_joke_found']} "
            f"actual={actual['no_joke_found']}"
        )

    # Blank title + body_internal is the convention for "generate any title — don't check source"
    title_source_is_wildcard = (
        not expected["title"] and expected["title_source"] == "body_internal"
    )
    # Title-Source may list multiple acceptable values separated by "|"
    acceptable_sources = set(expected["title_source"].split("|"))
    if not title_source_is_wildcard and actual["title_source"] not in acceptable_sources:
        failures.append(
            f"Title-Source mismatch: expected={expected['title_source']!r} "
            f"actual={actual['title_source']!r}"
        )

    # Body check only when a joke is expected
    if not expected["no_joke_found"]:
        exp_norm = _normalize_body(expected["joke_body"])
        act_norm = _normalize_body(actual["joke_body"])
        exp_tokens = exp_norm.split()
        act_tokens = act_norm.split()
        exp_count = len(exp_tokens)
        act_count = len(act_tokens)

        # Containment: expected token sequence must appear within actual tokens
        if exp_norm and exp_norm not in act_norm:
            failures.append(
                f"Body containment failed: expected tokens not found in extracted body\n"
                f"  expected ({exp_count} tokens): {exp_norm[:120]}{'...' if len(exp_norm) > 120 else ''}\n"
                f"  actual   ({act_count} tokens): {act_norm[:120]}{'...' if len(act_norm) > 120 else ''}"
            )

        # Bloat: extracted must not exceed expected by more than max(10, 10%)
        bloat_limit = max(10, int(exp_count * 0.10))
        if act_count > exp_count + bloat_limit:
            failures.append(
                f"Body bloat: extracted {act_count} tokens vs expected {exp_count} "
                f"(limit: +{bloat_limit})"
            )

    return failures


def _body_diff(expected_body: str, actual_body: str) -> str:
    """Return a unified diff of normalized bodies."""
    exp_lines = _normalize_body(expected_body).split()
    act_lines = _normalize_body(actual_body).split()
    # Chunk into ~10-token lines for readable diff
    def chunk(tokens, n=10):
        return [" ".join(tokens[i:i+n]) for i in range(0, len(tokens), n)]
    diff = difflib.unified_diff(
        chunk(exp_lines), chunk(act_lines),
        fromfile="expected (normalized)", tofile="actual (normalized)",
        lineterm="",
    )
    return "\n".join(diff)


# ---------------------------------------------------------------------------
# Per-case runner
# ---------------------------------------------------------------------------

def run_case(case_dir: Path) -> tuple[bool, str]:
    """
    Run evaluation for one fixture case.
    Returns (passed, detail_text).
    """
    stage1_path = case_dir / "stage1.json"
    expected_path = case_dir / "expected.txt"
    actual_path = case_dir / "actual-stage2.txt"

    if not stage1_path.exists():
        return False, "SKIP — stage1.json missing (run update_fixtures.py)"
    if not expected_path.exists():
        return False, "SKIP — expected.txt missing (write ground truth manually)"

    # Load stage1
    data = json.loads(stage1_path.read_text(encoding="utf-8"))
    segments = data["segments"]
    subject = data["subject"]

    # Call LLM
    try:
        result = extract_joke(segments, subject)
    except Exception as exc:
        return False, f"ERROR — extract_joke raised: {exc}"

    # Write actual-stage2.txt
    actual_text = _format_result_file(
        title=result.title,
        title_source=result.title_source,
        no_joke_found=result.no_joke_found,
        joke_body=result.joke_body,
    )
    actual_path.write_text(actual_text, encoding="utf-8")

    # Parse both files for comparison
    actual = _parse_result_file(actual_path)
    expected = _parse_result_file(expected_path)

    failures = _compare(expected, actual)
    if not failures:
        return True, ""

    detail_lines = []
    for f in failures:
        detail_lines.append(f"  FAIL: {f}")

    # Show body diff on body failures
    if any("Body" in f for f in failures):
        diff = _body_diff(expected["joke_body"], actual["joke_body"])
        if diff:
            detail_lines.append("  --- body diff ---")
            for line in diff.splitlines():
                detail_lines.append(f"  {line}")
        detail_lines.append(
            f"  word counts — expected: {len(expected['joke_body'].split())}  "
            f"actual: {len(actual['joke_body'].split())}"
        )

    return False, "\n".join(detail_lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    # Allow running against specific case dirs passed as args
    if len(sys.argv) > 1:
        case_dirs = [Path(p) for p in sys.argv[1:]]
    else:
        if not FIXTURES_DIR.exists():
            print(f"fixtures dir not found: {FIXTURES_DIR}")
            sys.exit(1)
        case_dirs = sorted(p for p in FIXTURES_DIR.iterdir() if p.is_dir())

    if not case_dirs:
        print("No fixture cases found.")
        sys.exit(0)

    results = []
    name_width = max(len(c.name) for c in case_dirs)

    for case_dir in case_dirs:
        passed, detail = run_case(case_dir)
        results.append((case_dir.name, passed, detail))

    print()
    print(f"{'Case':<{name_width}}  Result")
    print("-" * (name_width + 10))
    any_fail = False
    for name, passed, detail in results:
        status = "PASS" if passed else "FAIL"
        print(f"{name:<{name_width}}  {status}")
        if detail and not detail.startswith("SKIP"):
            print(detail)
        if not passed and not detail.startswith("SKIP"):
            any_fail = True
    print()

    total = len(results)
    skipped = sum(1 for _, _, d in results if d.startswith("SKIP"))
    ran = total - skipped
    passed_count = sum(1 for _, p, d in results if p and not d.startswith("SKIP"))

    if ran == 0:
        print(f"No cases ran ({skipped} skipped).")
    else:
        pct = int(100 * passed_count / ran)
        print(f"{passed_count}/{ran} passed ({pct}%)", end="")
        if skipped:
            print(f"  [{skipped} skipped]", end="")
        print()

    if any_fail:
        sys.exit(1)


if __name__ == "__main__":
    main()
