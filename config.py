OLLAMA_MODEL = "llama3.1:8b"
OLLAMA_CTX_MIN = 4096    # minimum context window (covers short jokes + full schema)
OLLAMA_MODEL_CTX = 131072  # maximum context window for this model
OLLAMA_HOST = "http://localhost:11434"
OLLAMA_TIMEOUT = 300  # seconds
OLLAMA_KEEP_ALIVE = 0  # unload model from memory immediately after each request

CONFIDENCE_THRESHOLD_BODY = 0.6
CONFIDENCE_THRESHOLD_TITLE = 0.6

SEP_MARKER = "<<<SEP>>>"
MIN_BODY_LENGTH = 20  # chars; shorter extracted bodies trigger needs_review

LOG_FILE = "logs/extraction.log"
LOG_LEVEL = "DEBUG"
