import os
import logging
from datetime import datetime
import structlog


# ======================================================
# DEBUG FLAG (from .env)
# ======================================================
DEBUG = os.getenv("DEBUG", "false").lower() == "true"


# ======================================================
# CUSTOM FILTER (CORE LOGIC)
# ======================================================
class DebugFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:

        message = record.getMessage()

        # ALWAYS allow warnings/errors
        if record.levelno >= logging.WARNING:
            return True

        # ALWAYS allow key business logs
        important_keywords = [
            "agent_request_started",
            "agent_request_completed",
            "response_quality",
            "bad_response_detected",
        ]

        if any(k in message for k in important_keywords):
            return True

        # If DEBUG mode → allow everything
        if DEBUG:
            return True

        # Drop noisy logs in production
        noisy_keywords = [
            "reason_node",
            "tool_node",
            "parsed",
            "filters",
            "state",
            "documents",
        ]

        if any(k in message for k in noisy_keywords):
            return False

        return True


class CustomLogger:
    """
    Structured JSON logger with DEBUG-aware filtering.
    """

    def __init__(self, log_dir: str = "logs"):

        self.logs_dir = os.path.join(os.getcwd(), log_dir)
        os.makedirs(self.logs_dir, exist_ok=True)

        log_file = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"
        self.log_file_path = os.path.join(self.logs_dir, log_file)

    def get_logger(self, name: str = __file__):

        logger_name = os.path.basename(name)

        # ======================================================
        # LOG LEVEL CONTROL
        # ======================================================
        log_level = logging.DEBUG if DEBUG else logging.INFO

        # -----------------------
        # File handler
        # -----------------------
        file_handler = logging.FileHandler(self.log_file_path)
        file_handler.setLevel(log_level)
        file_handler.setFormatter(logging.Formatter("%(message)s"))
        file_handler.addFilter(DebugFilter())   

        # -----------------------
        # Console handler
        # -----------------------
        console_handler = logging.StreamHandler()
        console_handler.setLevel(log_level)
        console_handler.setFormatter(logging.Formatter("%(message)s"))
        console_handler.addFilter(DebugFilter())   

        # -----------------------
        # Root logger setup
        # -----------------------
        root_logger = logging.getLogger()
        root_logger.setLevel(log_level)

        # Prevent duplicate handlers
        if not root_logger.handlers:
            root_logger.addHandler(console_handler)
            root_logger.addHandler(file_handler)

        # ======================================================
        # Structlog config
        # ======================================================
        structlog.configure(
            processors=[
                structlog.processors.TimeStamper(fmt="iso", utc=True, key="timestamp"),
                structlog.processors.add_log_level,
                structlog.processors.EventRenamer(to="event"),
                structlog.processors.JSONRenderer(),
            ],
            logger_factory=structlog.stdlib.LoggerFactory(),
            cache_logger_on_first_use=True,
        )

        return structlog.get_logger(logger_name)