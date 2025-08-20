# EEG-Based Classification of Alzheimer's Disease

An optimized machine learning pipeline that implements advanced signal processing techniques and XGBoost classification to distinguish between Alzheimer's Disease (AD) and healthy controls using electroencephalography (EEG) data.

## Classification Results
The hyperparameter-tuned XGBoost model, evaluated using Leave-One-Subject-Out (LOSO) cross-validation, achieved a mean **accuracy of 84.62%**, indicating robust subject-independent classification of the EEG data.

## Pipeline Overview

### 1. Signal Preprocessing
- **Bandpass Filtering**: 0.5–45 Hz Butterworth filter for noise removal
- **Artifact Subspace Reconstruction (ASR)**: Automatic removal of high-amplitude artifacts
- **Independent Component Analysis (ICA)**: Elimination of eye blink and muscle artifacts using ICLabel

**Raw EEG Data**
![Raw EEG](reports/figures/preprocessing/raw_eeg.png)

**Preprocessed EEG Data**
![Preprocessed EEG](reports/figures/preprocessing/preprocessed_eeg.png)

### 2. Feature Extraction
- **Epoching**: 4-second windows with 50% overlap for temporal analysis
- **Power Spectral Density**: Welch's method with 2-second windows for robust frequency domain analysis
- **Per-Channel Relative Band Power (RBP)**: Channel-specific normalized power across physiological frequency bands:
  - Delta (0.5–4 Hz), Theta (4–8 Hz), Alpha (8–13 Hz), Beta (13–25 Hz), Gamma (25–45 Hz)

### 3. Model Optimization
- **XGBoost Classifier**: Gradient boosting algorithm optimized for binary classification
- **Hyperparameter Tuning**: Optuna-based optimization using Leave-One-Subject-Out (LOSO) cross-validation
- **Performance Validation**: Rigorous cross-validation for reliable performance estimation

## Project Structure

```
eeg-classification-ad/
├── data/
│   ├── raw/                            # OpenNeuro ds004504 dataset
│   ├── interim/
│   │   └── epochs/                     # Preprocessed epochs (.fif files)
│   └── processed/                      # Feature-extracted datasets (.csv/.pkl)
├── src/
│   ├── pipeline.py                     # Main execution pipeline
│   ├── config.py                       # Configuration and logging setup
│   ├── models/
│   │   ├── xgboost.py                  # XGBoost implementation with Optuna
│   │   └── optimized_hyperparams.json  # Saved XGBoost optimal hyperparameters
│   └── utils/
│       └── dataset.py                  # EEG preprocessing and feature extraction
├── reports/
│   └── figures/
│       ├── preprocessing/              # Raw and preprocessed EEG plots
│       └── optuna/                     # Hyperparameter optimization plots
└── environment.yml                     # Conda environment specification
```

## Example Pipeline Logging

```
8:38:51 - INFO - Creating dataset (this may take a while)...
Preprocessing      : 100%|██████████████████████████████████████████| 65/65 [55:33<00:00, 51.29s/it]
Feature Extraction : 100%|██████████████████████████████████████████| 65/65 [00:59<00:00,  1.10it/s]
9:35:24 - INFO - Successfully created dataset.
9:35:24 - INFO - Optimizing XGBoost hyperparameters (this may take a while)...
Best trial: 19. Best value: 0.846154: 100%|█████████████████████████| 50/50 [09:48<00:00, 11.76s/it]
9:45:13 - INFO - Hyperparameter optimization complete.
9:45:13 - INFO - Saving optimized hyperparameters...
9:45:13 - INFO - Saving complete.
9:45:13 - INFO - Saving hyperparameter optimization plots...
9:45:14 - INFO - Saving complete.
```

## Quick Start

1. **Setup Environment**:
   ```
   conda env create -f environment.yml
   conda activate eeg-classification-ad
   ```

2. **Prepare Data**: Download the OpenNeuro ds004504 dataset and place all `sub-0XX` folders and `participants.tsv` into the `data/raw/` directory. 

3. **Run Pipeline**:
   ```
   python src/pipeline.py
   ```

## References

- Dataset: [OpenNeuro ds004504](https://openneuro.org/datasets/ds004504)
- Data Descriptor: [10.3390/data8060095](https://doi.org/10.3390/data8060095)
- Original Study: [10.1109/ACCESS.2023.3294618](https://doi.org/10.1109/ACCESS.2023.3294618)
- ASRpy: [GitHub Repository](https://github.com/DiGyt/asrpy) (BSD-3-Clause License)
