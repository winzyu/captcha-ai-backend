# app/config/config.py
import os
from pathlib import Path

class Config:
    # Get the base directory (where backend folder is)
    BASE_DIR = Path(__file__).parent.parent.parent.parent
    
    # Flask settings
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'dev-key-replace-in-production'
    
    # Model settings
    MODEL_PATH = BASE_DIR.parent / 'captcha' / 'models' / 'run_20250209_212716' / 'best_model.pt'
    IMAGE_SIZE = (120, 240)  # Height, Width
    NUM_CHANNELS = 1  # Grayscale
    SEQUENCE_LENGTH = 5  # Length of CAPTCHA text
    NUM_CLASSES = 62  # 10 digits + 26 lowercase + 26 uppercase
    
    # API settings
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max file size
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
    
    # Data settings
    RAW_CAPTCHA_PATH = BASE_DIR.parent / 'captcha' / 'data' / 'raw'
    
    @classmethod
    def init_app(cls):
        """Print all important paths for debugging"""
        print("\nConfiguration Paths:")
        print(f"Base Directory: {cls.BASE_DIR.absolute()}")
        print(f"Model Path: {cls.MODEL_PATH.absolute()}")
        print(f"Raw CAPTCHA Path: {cls.RAW_CAPTCHA_PATH.absolute()}\n")
