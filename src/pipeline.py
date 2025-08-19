from config import Config
from models.xgboost import XGBoost
from utils.dataset import Dataset
from utils.visualize import Visualizer


def main():
    """Entry point for the EEG classification pipeline."""
    # Setup
    logger = Config.setup_logging()
    Config.create_directories()

    # Process raw EEG dataset
    dataset = Dataset()
    if Config.LOAD_PROCESSED_DATASET:
        logger.info("Loading processed dataset...")
        processed_dataset_pkl_path = dataset.get_processed_dataset_filepath()
        logger.info("Successfully loaded processed dataset.")
    else:
        logger.info("Creating dataset (this may take a while)...")
        dataset.validate_raw_dataset()
        dataset.create_dataset()
        processed_dataset_pkl_path = dataset.save_dataset()
        logger.info("Successfully created dataset.")

    # Visualize processed dataset
    if Config.VISUALIZE:
        logger.info("Visualizing processed dataset...")
        visualizer = Visualizer(processed_dataset_pkl_path)
        visualizer.plot_rbp_by_band_and_group()
        logger.info("Visualization complete.")

    # Evaluate model using cross-validation
    if Config.EVALUATE:
        logger.info(
            f"Evaluating XGBoost model using {Config.CV_STRATEGY.upper()} cross-validation (this may take a while)..."
        )
        xgboost_model = XGBoost(processed_dataset_pkl_path, Config.CV_STRATEGY)
        acc = xgboost_model.evaluate()
        logger.info(f"Evaluation complete. Model accuracy: {acc * 100:.2f}%")


if __name__ == "__main__":
    main()
