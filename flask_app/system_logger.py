"""Thread-safe system logger for Flask application."""
import threading
from datetime import datetime


class SystemLogger:
    """Thread-safe system logger for UI display."""

    def __init__(self, max_lines=1000):
        self.logs = []
        self.max_lines = max_lines
        self.lock = threading.Lock()

    def log(self, message, level="INFO"):
        """Add a log entry."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"[{timestamp}] [{level}] {message}"

        with self.lock:
            self.logs.append(log_entry)
            if len(self.logs) > self.max_lines:
                self.logs.pop(0)

        # Also print to console
        print(log_entry)

    def get_logs(self):
        """Get all logs as formatted string."""
        with self.lock:
            return "\n".join(self.logs)

    def get_logs_list(self):
        """Get all logs as list."""
        with self.lock:
            return self.logs.copy()

    def clear(self):
        """Clear all logs."""
        with self.lock:
            self.logs = []


# Initialize global logger
system_logger = SystemLogger()
