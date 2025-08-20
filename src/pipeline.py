from config import Config
from models.xgboost import XGBoost
from utils.dataset import Dataset


class Pipeline:
    def __init__(self):
        Config.create_directories()
        self.logger = Config.setup_logging()

    def process_raw_eeg_dataset(self):
        """Process the raw EEG dataset."""
        dataset = Dataset()
        if Config.LOAD_PROCESSED_DATASET:
            self.logger.info("Loading processed dataset...")
            self.processed_dataset_pkl_path = dataset.get_processed_dataset_filepath()
            self.logger.info("Successfully loaded processed dataset.")
        else:
            self.logger.info("Creating dataset (this may take a while)...")
            dataset.validate_raw_dataset()
            dataset.create_dataset()
            self.logger.info("Dataset creation complete.")
            self.logger.info("Saving processed dataset...")
            self.processed_dataset_pkl_path = dataset.save_dataset()
            self.logger.info("Saving complete.")

    def optimize_xgboost_hyperparameters(self):
        """Optimize hyperparameters for the XGBoost model."""
        self.xgboost_model = XGBoost(self.processed_dataset_pkl_path)
        if Config.LOAD_OPTIMIZED_HYPERPARAMS:
            self.logger.info("Loading optimized XGBoost hyperparameters...")
            self.xgboost_model.load_optimized_hyperparams()
            self.logger.info("Successfully loaded optimized hyperparameters.")
        else:
            self.logger.info("Optimizing XGBoost hyperparameters (this may take a while)...")
            self.xgboost_model.optimize_hyperparams()
            self.logger.info("Hyperparameter optimization complete.")
            self.logger.info("Saving optimized hyperparameters...")
            self.xgboost_model.save_optimized_hyperparams()
            self.logger.info("Saving complete.")
            self.logger.info("Saving hyperparameter optimization plots...")
            self.xgboost_model.save_optuna_plots()
            self.logger.info("Saving complete.")


def main():
    """Main function to run the pipeline."""
    pipeline = Pipeline()
    pipeline.process_raw_eeg_dataset()
    pipeline.optimize_xgboost_hyperparameters()


if __name__ == "__main__":
    main()
