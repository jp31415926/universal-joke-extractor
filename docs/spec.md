# Joke Extraction System — Specification

## Objective

Process raw `.eml` email submissions to extract structured joke data. Target ~80% extraction accuracy across thousands of emails.

## Architecture

```
python extract.py email.eml
  → Stage 1: preprocess.py  (deterministic MIME cleaning)
  → Stage 2: llm.py         (Ollama structured extraction)
  → JSON to stdout
```

## Stage 1 — Deterministic Preprocessing (`preprocess.py`)

**MIME handling**
- Parse with Python stdlib `email` package
- Collect all `text/plain` parts; fall back to `text/html` (converted via `html2text`) if none
- Retain only `From:` and `Subject:` headers; discard all others

**Decoding**
- Decode quoted-printable via `email` stdlib
- Normalize charset to ISO-8859-1 (`errors='replace'`)

**Normalization (per segment)**
- Line endings → `\n`; collapse 3+ blank lines to 2
- Unicode smart quotes/apostrophes/dashes/ellipsis → ASCII equivalents
- Soft hyphens → removed; non-breaking spaces → space

**Quote & forward cleanup**
- Strip leading `>` markers line-by-line
- Remove inline forwarding headers (`From:`, `To:`, `Date:`, `Sent:`, `Subject:`) appearing mid-body
- Replace separator lines (`====`, `----`, `****`, 4+ chars) with `<<<SEP>>>`

**Signature & footer removal**
- Cut everything after `-- ` (standard sig delimiter)
- Strip trailing lines matching footer phrases: `sent from my`, `get outlook`, `unsubscribe`, `this email was sent`, etc.
- Strip trailing all-caps lines ≥20 chars (banner/footer heuristic)
- Leave ambiguous content for the LLM

**Output:** `PreprocessResult(segments: list[str], subject: str, from_addr: str)`
All segments passed to Stage 2; LLM selects the best one.

## Stage 2 — LLM Extraction (`llm.py`)

**Integration**
- Library: `ollama` Python package
- Model: `llama3.1:8b` (configurable)
- Host: `http://localhost:11434` (configurable)
- Structured output enforced via `format=<json_schema>`
- Temperature: 0.1

**LLM tasks**
1. Extract joke body verbatim (strip surrounding commentary)
2. Select best title — prefer body-internal title → subject line → generate one
3. Self-report confidence scores 0.0–1.0 for body and title
4. Set `no_joke_found: true` if email contains no joke

**LLM output schema**
```json
{
  "joke_body": "string",
  "title": "string",
  "title_source": "body_internal | subject | generated",
  "confidence": { "body": 0.0, "title": 0.0 },
  "no_joke_found": false
}
```

## Output Schema

**Success**
```json
{
  "status": "success",
  "from": "submitter@example.com",
  "title": "The Talking Dog",
  "title_source": "subject",
  "body": "A man walks into a bar...",
  "confidence": { "body": 0.85, "title": 0.92 },
  "needs_review": false
}
```

**Failure**
```json
{
  "status": "failure",
  "message": "reason"
}
```

`needs_review: true` when `confidence.body < CONFIDENCE_THRESHOLD_BODY`, `confidence.title < CONFIDENCE_THRESHOLD_TITLE`, or `len(body) < MIN_BODY_LENGTH`.

## Configuration (`config.py`)

| Key | Default | Description |
|---|---|---|
| `OLLAMA_MODEL` | `"llama3.1:8b"` | Ollama model name |
| `OLLAMA_HOST` | `"http://localhost:11434"` | Ollama endpoint |
| `CONFIDENCE_THRESHOLD_BODY` | `0.6` | Below this → `needs_review` |
| `CONFIDENCE_THRESHOLD_TITLE` | `0.6` | Below this → `needs_review` |
| `MIN_BODY_LENGTH` | `20` | Chars; shorter body → `needs_review` |
| `SEP_MARKER` | `"<<<SEP>>>"` | Separator replacement token |
| `LOG_FILE` | `"logs/extraction.log"` | Log file path |
| `LOG_LEVEL` | `"INFO"` | Python logging level |

## Error Handling

All errors emit failure JSON to stdout and exit 0. The pipeline never stops on a single bad email.

| Scenario | Behavior |
|---|---|
| File not found | failure JSON |
| MIME parse error | failure JSON |
| Ollama unreachable | failure JSON |
| Malformed LLM JSON | failure JSON |
| `no_joke_found: true` | failure JSON |
| Low confidence | success JSON + `needs_review: true` |

## Logging

Every email logged to `logs/extraction.log`:
- Timestamp, file path, `From`, `Subject`
- Segment count, total cleaned char count
- Extracted title, title source, body preview (≤200 chars)
- Confidence scores, `needs_review` flag
- Full tracebacks on errors

## Dependencies

```
ollama>=0.3.0
html2text>=2024.2.26
```

All other functionality uses Python 3.11+ stdlib.

## Testing Plan

**Stage 1 unit tests** (crafted `.eml` fixtures):
- Plain text email → correct `From`/`Subject` extraction
- HTML-only email → `html2text` conversion correct
- Quoted-printable email → proper decoding
- Forwarded chain → all segments extracted, quote markers stripped
- Email with `-- ` delimiter → signature removed
- Separator blocks → replaced with `<<<SEP>>>`

**Stage 2 integration tests** (real emails, ground truth known):
- Joke body matches expected output
- `title_source` correctly identified
- `needs_review` set/unset correctly per thresholds

**Error path tests:**
- Non-existent file → failure JSON, exit 0
- No joke content → `no_joke_found` → failure JSON
- Ollama down → failure JSON, exit 0

**Threshold tuning:** Run against labeled sample set; adjust `CONFIDENCE_THRESHOLD_BODY/TITLE` until manual review rate is acceptable.
