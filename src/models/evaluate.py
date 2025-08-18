import os
from itertools import combinations

import lightgbm as lgb
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

from config import Config


class Evaluator:
    def __init__(self, processed_dataset_pkl_path):
        try:
            self.df = pd.read_pickle(processed_dataset_pkl_path)
        except Exception as e:
            raise RuntimeError(
                f"Failed to load data from {processed_dataset_pkl_path}: {str(e)}"
            ) from e

        if "group" not in self.df.columns:
            raise ValueError("Processed DataFrame is missing 'group' column.")

        self.ad_cn_df = self.df[self.df["group"].isin(["A", "C"])]
        self.ftd_cn_df = self.df[self.df["group"].isin(["F", "C"])]
        if self.ad_cn_df.empty or self.ftd_cn_df.empty:
            raise ValueError("AD+CN or FTD+CN DataFrame is empty.")

        try:
            self.models = {
                "LightGBM": lgb.LGBMClassifier(random_state=42),
                "SVM": SVC(kernel="poly", random_state=42),
                "KNN": KNeighborsClassifier(n_neighbors=3),
                "MLP": MLPClassifier((3,), random_state=42),
                "Random Forest": RandomForestClassifier(random_state=42),
            }
        except Exception as e:
            raise RuntimeError(f"Failed to initialize models: {str(e)}") from e

    def evaluate_models(self, groups):
        """Evaluate models using every combination of feature subsets for the specified groups (ad_cn or ftd_cn)."""
        # Validate input
        if groups not in ["ad_cn", "ftd_cn"]:
            raise ValueError(f"Invalid group: {groups}. Expected 'ad_cn' or 'ftd_cn'.")
        df = self.ad_cn_df if groups == "ad_cn" else self.ftd_cn_df

        # Extract features and target from DataFrame
        features = df.drop(["subject_id", "group"], axis=1).columns
        y = df["group"]

        # Initialize DataFrame to store features and models
        features_models_df = pd.DataFrame().rename_axis("Features")

        try:
            # Iterate through every combination of feature subsets
            for k in range(1, len(features) + 1):
                for c in combinations(features, k):
                    X = df[list(c)]
                    # Perform cross-validation for each model
                    for name, model in self.models.items():
                        scores = cross_val_score(model, X, y, cv=None)
                        print(f"Completed: {c}, {name}")
                        features_models_df.loc[", ".join(c), name] = (
                            f"{scores.mean():.4f}"
                        )
        except Exception as e:
            raise RuntimeError(f"Failed to perform model evaluation: {str(e)}") from e

        # Save the features and models DataFrame
        try:
            filepath_csv = os.path.join(
                Config.INTERIM_DATA_DIR, f"{groups}_optimal_features_and_model.csv"
            )
            features_models_df.to_csv(filepath_csv, sep="\t")
        except Exception as e:
            raise RuntimeError(
                f"Failed to save optimal features and model DataFrame: {str(e)}"
            ) from e

        # Find the optimal features subset and model (max accuracy)
        max_col = features_models_df.max().idxmax()
        max_row = features_models_df[max_col].idxmax()
        max_val = features_models_df.loc[max_row, max_col]
        print(
            f"Optimal Features and Model for {groups.upper()} Classification:\nFeatures: {max_row}\nModel: {max_col}\nAccuracy: {max_val}"
        )
