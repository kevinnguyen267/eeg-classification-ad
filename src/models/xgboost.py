import pandas as pd
from sklearn.model_selection import LeaveOneOut, StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier


class XGBoost:
    def __init__(self, processed_dataset_pkl_path, cv_strategy):
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

        # Initialization
        self.model = XGBClassifier(objective="binary:logistic", random_state=42)
        if cv_strategy == "loso":
            self.cv_strategy = LeaveOneOut()
        elif cv_strategy == "stratified_kfold":
            self.cv_strategy = StratifiedKFold(
                n_splits=5, shuffle=True, random_state=42
            )
        else:
            raise ValueError(
                "Invalid cross-validation strategy. Use 'loso' or 'stratified_kfold'."
            )

    def evaluate(self):
        """Evaluate the model using cross-validation."""
        try:
            # Extract features and labels
            X = self.df.drop(["subject_id", "group"], axis=1)
            label_encoder = LabelEncoder()
            y = label_encoder.fit_transform(self.df["group"])

            # Perform cross-validation
            scores = cross_val_score(self.model, X, y, cv=self.cv_strategy)
            return scores.mean()

        except Exception as e:
            raise RuntimeError(f"Model evaluation failed: {str(e)}") from e
