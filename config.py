OLLAMA_MODEL = "llama3.1:8b"
OLLAMA_HOST = "http://localhost:11434"
OLLAMA_TIMEOUT = 300  # seconds

CONFIDENCE_THRESHOLD_BODY = 0.6
CONFIDENCE_THRESHOLD_TITLE = 0.6

SEP_MARKER = "<<<SEP>>>"
MIN_BODY_LENGTH = 20  # chars; shorter extracted bodies trigger needs_review

LOG_FILE = "logs/extraction.log"
LOG_LEVEL = "DEBUG"
