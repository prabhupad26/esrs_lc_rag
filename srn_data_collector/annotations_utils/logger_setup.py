# logger_setup.py

import logging


class ColoredFormatter(logging.Formatter):
    COLORS = {"ERROR": "\033[91m", "CRITICAL": "\033[91m"}  # Red  # Red

    RESET = "\033[0m"

    def format(self, record):
        log_message = super().format(record)
        if record.levelname in self.COLORS:
            return f"{self.COLORS[record.levelname]}{log_message}{self.RESET}"
        return log_message


def setup_logger(log_file="app.log", log_level=logging.DEBUG):
    # Create a logger
    logger = logging.getLogger(__name__)

    # Set the logging level
    logger.setLevel(log_level)

    # Create a file handler and set the logging level
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(log_level)

    # Create a console handler and set the logging level
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)

    # Create a formatter and attach it to the handlers
    formatter = ColoredFormatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # Add the handlers to the logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger
