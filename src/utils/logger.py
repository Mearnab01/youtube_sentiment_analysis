import os
import logging
from datetime import datetime

def get_logger(module_name):
    # Setup log directory
    log_dir = os.path.join(os.getcwd(), "logs")
    os.makedirs(log_dir, exist_ok=True)

    # Filename: e.g., 05_15_2026.log
    log_filename = f"{datetime.now().strftime('%m_%d_%Y')}.log"
    log_filepath = os.path.join(log_dir, log_filename)

    logger = logging.getLogger(module_name)
    logger.setLevel(logging.DEBUG)

    formatter = logging.Formatter("[ %(asctime)s ] %(name)s - %(levelname)s - %(message)s")

    # Avoid duplicate handlers if the logger is called multiple times
    if not logger.hasHandlers():
        # File Handler (Captures everything)
        file_handler = logging.FileHandler(log_filepath)
        file_handler.setFormatter(formatter)
        file_handler.setLevel(logging.DEBUG)

        # Console Handler (Only shows important stuff)
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        console_handler.setLevel(logging.INFO)

        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

    return logger