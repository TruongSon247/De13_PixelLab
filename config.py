import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

class Config:
    SECRET_KEY = os.environ.get("SECRET_KEY", "local-image-lab-change-me")
    MAX_CONTENT_LENGTH = 20 * 1024 * 1024
    MAX_IMAGE_PIXELS = 16_000_000
    MAX_IMAGE_DIMENSION = 10_000
    DEBUG = os.environ.get("FLASK_DEBUG", "0") == "1"
