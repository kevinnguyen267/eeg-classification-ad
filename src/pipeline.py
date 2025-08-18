from config import Config
from models.evaluate import Evaluator
from utils.dataset import Dataset
from utils.visualizer import Visualizer


def main():
    """Entry point for the EEG classification pipeline."""
    # Create necessary directories (if they do not exist)
    Config.create_directories()

    # Process raw EEG dataset
    dataset = Dataset()
    if Config.LOAD_PROCESSED_DATASET:
        processed_dataset_pkl_path = dataset.get_processed_dataset_filepath()
    else:
        dataset.validate_raw_dataset()
        dataset.create_dataset()
        processed_dataset_pkl_path = dataset.save_dataset()

    # Visualize processed dataset
    if Config.CREATE_AND_SAVE_FIGURES:
        visualizer = Visualizer(processed_dataset_pkl_path)
        visualizer.plot_rbp_by_band_and_group()

    # Evaluate features and models
    if Config.EVALUATE:
        evaluator = Evaluator(processed_dataset_pkl_path)
        evaluator.evaluate_models("ad_cn")
        evaluator.evaluate_models("ftd_cn")


if __name__ == "__main__":
    main()
