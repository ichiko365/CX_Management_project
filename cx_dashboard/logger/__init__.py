import logging
import os
import sys
from logging.handlers import RotatingFileHandler
from from_root import from_root
from datetime import datetime

# --- Constants for Log Configuration (Completed) ---
LOG_DIR = 'logs'
# The log file should have a static name for rotation to work.
LOG_FILE = 'app.log'
# Set maximum log file size to 5 MB
MAX_LOG_SIZE_BYTES = 5 * 1024 * 1024
# Keep up to 3 backup log files
BACKUP_COUNT = 3

# --- Path Construction ---
# Use from_root() to ensure the 'logs' directory is in your project root
log_dir_path = os.path.join(from_root(), LOG_DIR)
os.makedirs(log_dir_path, exist_ok=True)
log_file_path = os.path.join(log_dir_path, LOG_FILE)


# --- Configure the Root Logger ---
# We configure the root logger, and other modules will inherit this setup.
logging.basicConfig(
    level=logging.DEBUG, # Set the lowest level for the root logger
    format="[ %(asctime)s ] %(name)s - %(levelname)s - %(message)s",
    handlers=[
        # Handler 1: Writes DEBUG level logs and above to a rotating file
        RotatingFileHandler(log_file_path, maxBytes=MAX_LOG_SIZE_BYTES, backupCount=BACKUP_COUNT),
        
        # Handler 2: Writes INFO level logs and above to the console
        logging.StreamHandler(sys.stdout)
    ]
)

# You can optionally adjust the console handler's level if you want less verbose output there
# This is done by getting the specific handler after basicConfig
for handler in logging.getLogger().handlers:
    if isinstance(handler, logging.StreamHandler):
        handler.setLevel(logging.INFO)

# Create a logger instance to be imported by other modules if needed,
# though using logging.getLogger(__name__) is the standard practice.
logger = logging.getLogger("CXDashboardLogger")