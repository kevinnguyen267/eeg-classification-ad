import os
from pathlib import Path


class Config:
    # Directories
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    DATA_DIR = PROJECT_ROOT / "data"
    RAW_DATA_DIR = DATA_DIR / "raw"
    INTERIM_DATA_DIR = DATA_DIR / "interim"
    PROCESSED_DATA_DIR = DATA_DIR / "processed"
    REPORTS_DIR = PROJECT_ROOT / "reports"
    FIGURES_DIR = REPORTS_DIR / "figures"

    # Load from checkpoints
    LOAD_PREPROCESSED_EPOCHS = False
    LOAD_PROCESSED_DATASET = False

    # Create and save figures
    CREATE_AND_SAVE_FIGURES = False

    # Evaluate features and models
    EVALUATE = False

    @staticmethod
    def create_directories():
        """Create necessary directories if they do not exist"""
        os.makedirs(Config.INTERIM_DATA_DIR, exist_ok=True)
        os.makedirs(os.path.join(Config.INTERIM_DATA_DIR, "epochs"), exist_ok=True)
        os.makedirs(Config.PROCESSED_DATA_DIR, exist_ok=True)
        os.makedirs(Config.REPORTS_DIR, exist_ok=True)
        os.makedirs(Config.FIGURES_DIR, exist_ok=True)
