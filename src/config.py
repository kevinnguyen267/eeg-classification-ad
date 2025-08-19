import logging
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

    # Load dataset from checkpoints
    LOAD_PREPROCESSED_EPOCHS = True
    LOAD_PROCESSED_DATASET = True

    # Visualize processed dataset
    VISUALIZE = False

    # Evaluate model using cross-validation
    EVALUATE = True
    CV_STRATEGY = "stratified_kfold"  # "loso" or "stratified_kfold"

    @staticmethod
    def setup_logging():
        """Configure logging for the project."""
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
            datefmt="%H:%M:%S",
        )
        return logging.getLogger(__name__)

    @staticmethod
    def create_directories():
        """Create necessary directories if they do not exist."""
        os.makedirs(Config.INTERIM_DATA_DIR, exist_ok=True)
        os.makedirs(os.path.join(Config.INTERIM_DATA_DIR, "epochs"), exist_ok=True)
        os.makedirs(Config.PROCESSED_DATA_DIR, exist_ok=True)
        os.makedirs(Config.REPORTS_DIR, exist_ok=True)
        os.makedirs(Config.FIGURES_DIR, exist_ok=True)
