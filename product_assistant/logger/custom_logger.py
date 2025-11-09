import os
import logging
from datetime import datetime
import structlog


class CustomLogger:
    """
    A custom structured logging utility for Python applications.

    This class sets up a logger that outputs **JSON-formatted logs**
    to both the console and a timestamped log file. It uses the
    `structlog` library to ensure that logs are structured, human-readable,
    and machine-parsable for downstream processing (e.g., ELK, Datadog, CloudWatch).

    Features:
        - Creates a dedicated log directory (default: `./logs`)
        - Generates timestamped log files
        - Sends logs to both console and file in JSON format
        - Includes UTC timestamps, log levels, and event names

    Example:
        >>> logger = CustomLogger().get_logger(__name__)
        >>> logger.info("User login successful", user_id=123)
        >>> logger.error("File not found", path="/tmp/data.csv")
    """

    def __init__(self, log_dir: str = "logs"):
        """
        Initialize the custom logger and ensure log directory setup.

        Args:
            log_dir (str): Directory name for storing log files.
                           Defaults to "logs" inside the current working directory.
        """
        # Ensure the logs directory exists (create if missing)
        self.logs_dir = os.path.join(os.getcwd(), log_dir)
        os.makedirs(self.logs_dir, exist_ok=True)

        # Generate a timestamped log filename (e.g., 11_09_2025_16_40_00.log)
        log_file = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"
        self.log_file_path = os.path.join(self.logs_dir, log_file)

    def get_logger(self, name: str = __file__):
        """
        Create and configure a structured JSON logger.

        Args:
            name (str): Name of the logger (usually `__name__` or `__file__`).

        Returns:
            structlog.stdlib.BoundLogger: A structured logger instance.
        """
        logger_name = os.path.basename(name)

        # -----------------------
        # Setup log file handler
        # -----------------------
        file_handler = logging.FileHandler(self.log_file_path)
        file_handler.setLevel(logging.INFO)
        # Each log line will be raw JSON (structlog formats it)
        file_handler.setFormatter(logging.Formatter("%(message)s"))

        # -----------------------
        # Setup console handler
        # -----------------------
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(logging.Formatter("%(message)s"))

        # -----------------------
        # Base Python logging setup
        # -----------------------
        logging.basicConfig(
            level=logging.INFO,
            format="%(message)s",  # structlog handles JSON formatting
            handlers=[console_handler, file_handler],
        )

        # -----------------------
        # Structlog configuration
        # -----------------------
        structlog.configure(
            processors=[
                # Add UTC ISO8601 timestamps to logs
                structlog.processors.TimeStamper(fmt="iso", utc=True, key="timestamp"),
                # Include the log level (e.g., INFO, ERROR)
                structlog.processors.add_log_level,
                # Rename 'event' field for clarity in output
                structlog.processors.EventRenamer(to="event"),
                # Render final structured JSON
                structlog.processors.JSONRenderer(),
            ],
            logger_factory=structlog.stdlib.LoggerFactory(),
            cache_logger_on_first_use=True,
        )

        return structlog.get_logger(logger_name)
