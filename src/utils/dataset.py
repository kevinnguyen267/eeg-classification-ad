import os
import warnings
from glob import glob

import asrpy
import mne
import numpy as np
import pandas as pd
from mne_icalabel import label_components
from tqdm import tqdm

from config import Config


class Dataset:
    def __init__(self):
        pass

    def validate_raw_dataset(self):
        """Check if the raw data directory exists and contains the required files."""
        # Check if the raw data directory exists
        if not os.path.exists(Config.RAW_DATA_DIR):
            raise FileNotFoundError(
                f"Raw dataset directory {Config.RAW_DATA_DIR} does not exist. Please download the raw dataset and place it in the correct directory."
            )

        # Check if the required file 'participants.tsv' exists
        self.participants_path = os.path.join(Config.RAW_DATA_DIR, "participants.tsv")
        if not os.path.exists(self.participants_path):
            raise FileNotFoundError(
                f"Required file 'participants.tsv' not found in {Config.RAW_DATA_DIR}. Please ensure the dataset is complete."
            )

        # Check for required columns in 'participants.tsv'
        try:
            participants_df = pd.read_csv(self.participants_path, sep="\t")
            required_columns = ["participant_id", "Group"]
            missing_columns = [
                col for col in required_columns if col not in participants_df.columns
            ]
            if missing_columns:
                raise ValueError(
                    f"Missing required columns in participants.tsv: {missing_columns}"
                )

            # Create DataFrame
            self.participants_df = participants_df[["participant_id", "Group"]].rename(
                columns={"Group": "group", "participant_id": "subject_id"}
            )

        except Exception as e:
            raise RuntimeError(f"Failed to read participants.tsv: {str(e)}") from e

        # Check if the required .set files exist (ignore the derivatives folder)
        self.filepaths = glob(os.path.join(Config.RAW_DATA_DIR, "*", "*", "*.set"))
        if len(self.filepaths) == 0:
            raise FileNotFoundError(
                f"No .set files found in {Config.RAW_DATA_DIR}. Please ensure the dataset is complete."
            )

        # Filter out non-AD/CN subjects
        self.participants_df = self.participants_df[
            self.participants_df["group"].isin(["A", "C"])
        ]
        if self.participants_df.empty:
            raise ValueError(f"No AD or CN subjects found in {self.participants_path}.")
        subject_ids = set(self.participants_df["subject_id"])
        self.filepaths = [
            fp
            for fp in self.filepaths
            if os.path.basename(fp).removesuffix("_task-eyesclosed_eeg.set")
            in subject_ids
        ]
        if len(self.filepaths) == 0:
            raise FileNotFoundError(
                f"No AD or CN group .set files found in {Config.RAW_DATA_DIR}. Please ensure the dataset is complete."
            )

    def read_raw_eeg(self, filepath):
        """Read a raw EEG file."""
        try:
            # Input validation
            if not os.path.exists(filepath):
                raise FileNotFoundError(f"EEG file not found: {filepath}")
            ext = os.path.splitext(filepath)[1].lower()
            if not ext == ".set":
                raise ValueError(f"Expected .set file format, got: {ext}")

            # Load the raw EEG file
            raw = mne.io.read_raw_eeglab(filepath, preload=True, verbose=False)
            if raw is None:
                raise RuntimeError(f"Raw EEG data returned None: {filepath}")

            return raw

        except Exception as e:
            raise RuntimeError(f"Failed to read EEG file {filepath}: {str(e)}") from e

    def preprocess_eeg(self, raw):
        """Preprocess the raw EEG data with artifact removal."""
        try:
            # Input validation
            if raw is None:
                raise ValueError("Raw EEG data is None")
            if len(raw.info["ch_names"]) == 0:
                raise ValueError("No channels found in EEG data")
            if raw.info["sfreq"] <= 0:
                raise ValueError(f"Invalid sampling frequency: {raw.info['sfreq']}")

            # Bandpass filter (0.5-45 Hz)
            raw_for_ica = raw.copy()
            iir_params = dict(order=4, ftype="butter")
            raw_for_ica.filter(
                l_freq=0.5,
                h_freq=45,
                method="iir",
                iir_params=iir_params,
                verbose=False,
            )

            # ASR for high-amplitude artifacts
            asr = asrpy.ASR(sfreq=raw_for_ica.info["sfreq"], cutoff=17, win_len=0.5)
            asr.fit(raw_for_ica)
            raw_asr = asr.transform(raw_for_ica)

            # ICA for eye and muscle artifacts
            ica = mne.preprocessing.ICA(
                n_components=0.999999, method="infomax", fit_params=dict(extended=True)
            )
            ica.fit(raw_asr, verbose=False)

            # Suppress warnings since raw data is already re-referenced
            # and we're using 0.5-45 Hz instead of 1-100 Hz filtering to preserve information
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message=".*common average reference.*",
                    category=RuntimeWarning,
                )
                warnings.filterwarnings(
                    "ignore",
                    message=".*not filtered between 1 and 100 Hz.*",
                    category=RuntimeWarning,
                )
                labels = label_components(raw_asr, ica, "iclabel")

            labels_to_remove = ["eye blink", "muscle artifact"]
            ica.exclude = [
                i
                for i, label in enumerate(labels["labels"])
                if label in labels_to_remove
            ]
            raw_clean = ica.apply(raw_asr, verbose=False)

            return raw_clean

        except Exception as e:
            raise RuntimeError(f"EEG preprocessing failed: {str(e)}") from e

    def epoch_eeg(self, data):
        """Epoch the preprocessed EEG data into fixed-length segments."""
        try:
            # Input validation
            if data is None:
                raise ValueError("Preprocessed EEG data is None")

            # Create 4-second fixed-length events with 50% overlap
            epoch_duration = 4
            events = mne.make_fixed_length_events(
                data, duration=epoch_duration, overlap=0.5
            )

            if len(events) == 0:
                raise RuntimeError("No events could be created from the data")

            # Create epochs
            epochs = mne.Epochs(
                data,
                events,
                tmin=0,
                tmax=epoch_duration,
                baseline=None,
                preload=True,
                verbose=False,
            )

            return epochs

        except Exception as e:
            raise RuntimeError(f"Epoching failed: {str(e)}") from e

    def compute_rbps(self, subject_id, epochs, sfreq, win, bands):
        """Vectorized implementation to compute Relative Band Powers (RBPs) from epoched EEG data."""
        try:
            # Input validation
            if epochs is None or len(epochs) == 0:
                raise ValueError("Epochs object is empty or None")
            if sfreq <= 0:
                raise ValueError(f"Invalid sampling frequency: {sfreq}")
            if win <= 0:
                raise ValueError(f"Invalid window length: {win}")

            # Extract data array
            x = epochs.get_data()  # shape: (n_epochs, n_channels, n_times)
            if x.size == 0:
                raise ValueError("No data found in epochs")

            # Compute Power Spectral Density using Welch's method
            psds, freqs = mne.time_frequency.psd_array_welch(
                x,
                sfreq,
                fmin=0.5,
                fmax=45,
                n_fft=win,
                n_overlap=win // 2,
                verbose=False,
            )  # psds shape: (n_epochs, n_channels, n_freqs)
            if psds.size == 0 or freqs.size == 0:
                raise RuntimeError("PSD computation result is empty")

            # Sum across frequencies (axis=2) then take mean across epochs (axis=0) to get per channel total power
            # shape: (n_channels,)
            total_power_per_channel = np.sum(psds, axis=2).mean(axis=0)
            if np.any(total_power_per_channel <= 0):
                raise RuntimeError(
                    "Total power is non-positive for at least one channel"
                )

            # Create subject and frequency band masks for later assignment
            subject_mask = self.participants_df["subject_id"] == subject_id
            freq_masks = np.array(
                [(freqs >= fmin) & (freqs < fmax) for fmin, fmax in bands.values()]
            )  # shape: (n_bands, n_freqs)

            # psds * freq_masks = shape: (n_epochs, n_channels, n_bands, n_freqs)
            # Sum over frequencies (axis=3) and mean over epochs (axis=0)
            band_power_per_channel = np.sum(
                psds[:, :, None, :] * freq_masks[None, None, :, :], axis=3
            ).mean(axis=0)  # shape: (n_channels, n_bands)

            # Compute RBPs for all bands per channel
            rbp_per_channel = (
                band_power_per_channel / total_power_per_channel[:, None]
            )  # shape: (n_channels, n_bands)

            # Create column names and RBP values for assignment
            columns = []
            rbps = []
            band_names = list(bands.keys())
            for band_idx, band_name in enumerate(band_names):
                band_columns = [
                    f"{band_name}_rbp_ch{ch_idx + 1}"
                    for ch_idx in range(rbp_per_channel.shape[0])
                ]
                columns.extend(band_columns)
                rbps.extend(rbp_per_channel[:, band_idx])

            # Assignment to subject
            self.participants_df.loc[subject_mask, columns] = rbps

        except Exception as e:
            raise RuntimeError(f"Failed to compute RBPs: {str(e)}") from e

    def extract_features(self):
        """Extract features from the epoched EEG data."""
        try:
            # Get sampling frequency from first valid file
            sfreq = self.read_raw_eeg(self.filepaths[0]).info["sfreq"]
            if sfreq <= 0:
                raise ValueError(f"Invalid sampling frequency: {sfreq}")

            # 2-second window length for Welch's method
            win = int(2 * sfreq)

            # Define frequency bands and their corresponding ranges
            bands = {
                "delta": (0.5, 4),
                "theta": (4, 8),
                "alpha": (8, 13),
                "beta": (13, 25),
                "gamma": (25, 45),
            }

            # Extract Relative Band Powers (RBPs) from epochs
            filepaths = glob(
                os.path.join(Config.INTERIM_DATA_DIR, "epochs", "*_epo.fif")
            )
            for filepath in tqdm(filepaths, desc="Feature Extraction "):
                epochs = mne.read_epochs(filepath, verbose=False)
                subject_id = os.path.basename(filepath).rstrip("_epo.fif")
                self.compute_rbps(subject_id, epochs, sfreq, win, bands)

        except Exception as e:
            raise RuntimeError(f"Feature extraction failed: {str(e)}") from e

    def create_dataset(self):
        """Create a dataset from the validated raw files."""
        try:
            # Create epochs (if existing epochs aren't configured to be loaded from save)
            if not Config.LOAD_PREPROCESSED_EPOCHS:
                for filepath in tqdm(self.filepaths, desc="Preprocessing      "):
                    # Preprocess and epoch for each subject
                    subject_id = os.path.basename(filepath).split("_")[0]
                    raw = self.read_raw_eeg(filepath)
                    preprocessed = self.preprocess_eeg(raw)
                    epochs = self.epoch_eeg(preprocessed)
                    epochs.save(
                        os.path.join(
                            Config.INTERIM_DATA_DIR, "epochs", f"{subject_id}_epo.fif"
                        ),
                        overwrite=True,
                        verbose=False,
                    )

            # Extract features from epochs
            self.extract_features()

        except Exception as e:
            raise RuntimeError(f"Failed to create dataset: {str(e)}") from e

    def save_dataset(self):
        """Save the created dataset as a CSV and a pickle file."""
        try:
            filepath_csv = os.path.join(Config.PROCESSED_DATA_DIR, "participants.csv")
            filepath_pkl = os.path.join(Config.PROCESSED_DATA_DIR, "participants.pkl")
            self.participants_df.to_csv(filepath_csv, sep="\t")
            self.participants_df.to_pickle(filepath_pkl)
            return filepath_pkl
        except Exception as e:
            raise RuntimeError(f"Failed to save dataset: {str(e)}") from e

    def get_processed_dataset_filepath(self):
        """Get the filepath of the processed dataset."""
        filepath_pkl = os.path.join(Config.PROCESSED_DATA_DIR, "participants.pkl")
        if not os.path.exists(filepath_pkl):
            raise FileNotFoundError(
                f"Processed dataset not found at {filepath_pkl}. Please check that the dataset exists at this location, or update the configuration if you do not intend to load it."
            )
        return filepath_pkl
