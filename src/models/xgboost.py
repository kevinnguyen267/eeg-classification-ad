import json
import os
import warnings

import matplotlib.pyplot as plt
import optuna
import pandas as pd
from optuna.exceptions import ExperimentalWarning
from optuna.visualization.matplotlib import (
    plot_optimization_history,
    plot_parallel_coordinate,
    plot_param_importances,
)
from sklearn.model_selection import LeaveOneOut, cross_val_score
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

from config import Config


class XGBoost:
    def __init__(self, processed_dataset_pkl_path):
        # Suppress Optuna logging
        optuna.logging.set_verbosity(optuna.logging.WARNING)

        # Load the processed dataset
        try:
            self.df = pd.read_pickle(processed_dataset_pkl_path)
        except Exception as e:
            raise RuntimeError(
                f"Failed to load data from {processed_dataset_pkl_path}: {str(e)}"
            ) from e

        # Input validation
        if self.df.empty:
            raise ValueError("Processed DataFrame is empty.")
        if "subject_id" not in self.df.columns or "group" not in self.df.columns:
            raise ValueError(
                "Processed DataFrame is missing 'subject_id' or 'group' column."
            )

        # X: features matrix, y: encoded target labels
        self.X = self.df.drop(["subject_id", "group"], axis=1)
        self.label_encoder = LabelEncoder()
        self.y = self.label_encoder.fit_transform(self.df["group"])

        # Filepath for saving/loading the optimized hyperparameters
        self.optimized_hyperparams_filepath = os.path.join(
            Config.MODELS_DIR, "optimized_hyperparams.json"
        )

    def objective(self, trial):
        """Objective function for hyperparameter optimization."""
        hyperparams = {
            "max_depth": trial.suggest_int("max_depth", 2, 10),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 15),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "n_estimators": trial.suggest_int("n_estimators", 100, 500),
            "gamma": trial.suggest_float("gamma", 0, 0.5),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        }

        model = XGBClassifier(
            **hyperparams, objective="binary:logistic", random_state=42
        )

        return cross_val_score(
            model, self.X, self.y, cv=LeaveOneOut(), scoring="accuracy"
        ).mean()

    def optimize_hyperparams(self):
        """Optimize hyperparameters using Optuna."""
        self.study = optuna.create_study(
            direction="maximize", sampler=optuna.samplers.TPESampler(seed=42)
        )
        self.study.optimize(
            self.objective,
            n_trials=50,
            show_progress_bar=True,
        )

    def save_optimized_hyperparams(self):
        """Save the optimized hyperparameters to a JSON file."""
        try:
            with open(
                self.optimized_hyperparams_filepath,
                "w",
            ) as f:
                json.dump(self.study.best_params, f)
        except Exception as e:
            raise RuntimeError(
                f"Failed to save optimized hyperparameters: {str(e)}"
            ) from e

    def save_optuna_plots(self):
        """Save Optuna optimization plots."""
        if not hasattr(self, "study"):
            self.logger.warning("No Optuna study found. Skipping plot saving.")
            return
        try:
            # Types of plots
            plots = {
                "optimization_history": plot_optimization_history,
                "param_importances": plot_param_importances,
                "parallel_coordinate": plot_parallel_coordinate,
            }

            # Generate and save each plot
            for name, plot_func in plots.items():
                # Suppress warnings for experimental plots using Matplotlib
                with warnings.catch_warnings():
                    warnings.filterwarnings(
                        "ignore",
                        message=".*is experimental.*",
                        category=ExperimentalWarning,
                    )
                    fig = plot_func(self.study).figure
                fig.savefig(
                    os.path.join(Config.OPTUNA_DIR, f"{name}.png"),
                    dpi=300,
                    bbox_inches="tight",
                )
                plt.close(fig)
        except Exception as e:
            raise RuntimeError(f"Failed to save Optuna plots: {str(e)}") from e

    def load_optimized_hyperparams(self):
        """Load the optimized hyperparameters from a JSON file."""
        if not os.path.exists(self.optimized_hyperparams_filepath):
            raise FileNotFoundError(
                f"Optimized hyperparameters not found at {self.optimized_hyperparams_filepath}. Please check that the optimized hyperparameters exist at this location, or update the configuration if you do not intend to load it."
            )
        try:
            with open(self.optimized_hyperparams_filepath, "r") as f:
                hyperparams = json.load(f)
            self.model = XGBClassifier(**hyperparams, random_state=42)
        except Exception as e:
            raise RuntimeError(
                f"Failed to load optimized hyperparameters: {str(e)}"
            ) from e
