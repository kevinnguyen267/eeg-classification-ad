import os

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from config import Config


class Visualizer:
    def __init__(self, processed_dataset_pkl_path):
        try:
            self.df = pd.read_pickle(processed_dataset_pkl_path)
        except Exception as e:
            raise RuntimeError(
                f"Failed to load data from {processed_dataset_pkl_path}: {str(e)}"
            ) from e

        # Global plot settings
        mpl.style.use("fivethirtyeight")
        mpl.rcParams["figure.figsize"] = (10, 6)
        mpl.rcParams["figure.dpi"] = 100

    def plot_rbp_by_band_and_group(self):
        """Create a boxplot visualization of Relative Band Power (RBP) by frequency band and diagnostic group."""
        # Check for required columns
        required_columns = [
            "subject_id",
            "group",
            "delta_rbp",
            "theta_rbp",
            "alpha_rbp",
            "beta_rbp",
            "gamma_rbp",
        ]
        missing_columns = [
            col for col in required_columns if col not in self.df.columns
        ]
        if missing_columns:
            raise ValueError(
                f"Missing required columns in processed dataset: {missing_columns}"
            )

        try:
            # Detect RBP columns
            band_columns = [col for col in self.df.columns if "_rbp" in col]

            # Melt the DataFrame to long format
            df_melted = self.df.melt(
                id_vars=["group"],
                value_vars=band_columns,
                var_name="band",
                value_name="power",
            )

            # Clean group names
            df_melted["group"] = df_melted["group"].replace(
                {"A": "AD", "F": "FTD", "C": "CN"}
            )

            # Clean band names
            df_melted["band"] = df_melted["band"].str.replace("_rbp", "")

            # Create boxplot
            sns.boxplot(
                data=df_melted, x="band", y="power", hue="group", width=0.5, gap=0.1
            )
            plt.title("Relative Band Power (RBP) by Frequency Band and Group")
            plt.xlabel("Frequency Band")
            plt.ylabel("RBP")
            plt.legend(title="Group")
            plt.tight_layout()
            plt.savefig(
                os.path.join(Config.FIGURES_DIR, "rbp_by_band_and_group_boxplot"),
                bbox_inches="tight",
            )
        except Exception as e:
            raise RuntimeError(f"Failed to create or save boxplot: {str(e)}") from e
