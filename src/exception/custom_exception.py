import sys
import traceback
from typing import Optional, cast

class ProductAssistantException(Exception):
    """
    Custom exception class for the Product Assistant application.

    This class provides detailed contextual information about exceptions,
    including:
        - The file name where the error occurred
        - The line number of the error
        - The full traceback (if available)
    
    It can handle error details from:
        - Standard Python exceptions
        - `sys` module's `exc_info`
        - Arbitrary exception objects

    Example:
        >>> try:
        ...     1 / 0
        ... except Exception as e:
        ...     raise ProductAssistantException("Division error", e)
    """

    def __init__(self, error_message: str, error_details: Optional[object] = None):
        """
        Initialize the custom exception with a normalized error message and
        extract traceback details.

        Args:
            error_message (str | Exception): 
                The main message describing the error.
                Can be a string or another Exception object.
            error_details (Optional[object]): 
                Additional error context (can be `sys`, an Exception, or None).
        """
        # Normalize error message (ensure it's always a string)
        if isinstance(error_message, BaseException):
            norm_msg = str(error_message)
        else:
            norm_msg = str(error_message)

        # Initialize traceback components
        exc_type = exc_value = exc_tb = None

        # Determine the source of exception details
        if error_details is None:
            # Use the current active exception info, if available
            exc_type, exc_value, exc_tb = sys.exc_info()
        else:
            # If provided explicitly, extract info accordingly
            if hasattr(error_details, "exc_info"):  
                # Case 1: error_details is `sys` or a sys-like object
                exc_info_obj = cast(sys, error_details)
                exc_type, exc_value, exc_tb = exc_info_obj.exc_info()
            elif isinstance(error_details, BaseException):
                # Case 2: error_details is an Exception instance
                exc_type, exc_value, exc_tb = (
                    type(error_details),
                    error_details,
                    error_details.__traceback__,
                )
            else:
                # Fallback: use whatever is in the current context
                exc_type, exc_value, exc_tb = sys.exc_info()

        # Traverse to the last frame for the most relevant error location
        last_tb = exc_tb
        while last_tb and last_tb.tb_next:
            last_tb = last_tb.tb_next

        # Extract filename and line number where the error occurred
        self.file_name = last_tb.tb_frame.f_code.co_filename if last_tb else "<unknown>"
        self.lineno = last_tb.tb_lineno if last_tb else -1
        self.error_message = norm_msg

        # Build a formatted traceback string if available
        if exc_type and exc_tb:
            self.traceback_str = ''.join(traceback.format_exception(exc_type, exc_value, exc_tb))
        else:
            self.traceback_str = ""

        # Call the base Exception class constructor
        super().__init__(self.__str__())

    def __str__(self) -> str:
        """
        Return a compact, readable string representation of the error,
        including file, line, and message. Includes traceback if available.
        """
        base = f"Error in [{self.file_name}] at line [{self.lineno}] | Message: {self.error_message}"
        if self.traceback_str:
            return f"{base}\nTraceback:\n{self.traceback_str}"
        return base

    def __repr__(self) -> str:
        """
        Return a developer-friendly string representation of the exception,
        showing key attributes.
        """
        return (
            f"ProductAssistantException("
            f"file={self.file_name!r}, line={self.lineno}, message={self.error_message!r})"
        )
