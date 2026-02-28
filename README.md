# Universal Joke Extractor

Extracts structured joke data from raw `.eml` email submissions using deterministic preprocessing + a local LLM (Ollama). Designed to process thousands of old joke-submission emails with ~80% extraction accuracy.

## Setup

```bash
pip install -r requirements.txt
```

Requires [Ollama](https://ollama.com) running locally with a model pulled:

```bash
ollama pull llama3.1:8b
```

## Usage

```bash
python extract.py path/to/email.eml
```

Prints one JSON object to stdout. Logs details to `logs/extraction.log`.

## Output

**Success:**
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

**Failure:**
```json
{
  "status": "failure",
  "message": "LLM found no joke in email content"
}
```

`needs_review: true` is added to success output when confidence scores fall below the configured thresholds or the extracted body is very short.

`title_source` is one of `body_internal`, `subject`, or `generated`.

## Architecture

```
email.eml
  ↓
preprocess.py  — MIME parsing, charset decoding (ISO-8859-1), quote/sig/footer
                 stripping, separator normalization, HTML→text fallback
  ↓
llm.py         — Ollama structured JSON extraction; LLM selects best text
                 segment, extracts joke body verbatim, picks best title,
                 self-reports confidence scores
  ↓
stdout (JSON)
```

All MIME text parts are passed to the LLM labeled as segments — it picks the best one. This handles arbitrarily nested forwarding chains without heuristic depth guessing.

## Configuration

Edit `config.py` to change behavior:

| Key | Default | Description |
|---|---|---|
| `OLLAMA_MODEL` | `"llama3.1:8b"` | Any model available in your Ollama install |
| `OLLAMA_HOST` | `"http://localhost:11434"` | Ollama endpoint |
| `CONFIDENCE_THRESHOLD_BODY` | `0.6` | Below this triggers `needs_review` |
| `CONFIDENCE_THRESHOLD_TITLE` | `0.6` | Below this triggers `needs_review` |
| `MIN_BODY_LENGTH` | `20` | Chars; shorter extracted body triggers `needs_review` |
| `LOG_FILE` | `"logs/extraction.log"` | Log output path |

## Batch Processing

The script processes one email at a time. To run over a directory:

```bash
for f in emails/*.eml; do
  python extract.py "$f" >> results.jsonl
done
```

## Files

```
extract.py       — CLI entry point
preprocess.py    — Stage 1: deterministic preprocessing
llm.py           — Stage 2: Ollama LLM extraction
config.py        — Constants and thresholds
requirements.txt — pip dependencies
logs/            — Log output directory
spec.md          — Full developer specification
```
