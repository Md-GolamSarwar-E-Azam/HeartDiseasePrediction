import os
import sys
from dataclasses import dataclass

import pandas as pd

from src.exception import CustomException
from src.logger import logging


@dataclass
class DataValidationConfig:
    validation_status_file_path: str = os.path.join("artifacts", "data_validation_status.txt")


class DataValidation:
    def __init__(self):
        self.data_validation_config = DataValidationConfig()
        self.expected_columns = [
            "age",
            "sex",
            "cp",
            "trestbps",
            "chol",
            "fbs",
            "restecg",
            "thalach",
            "exang",
            "oldpeak",
            "slope",
            "ca",
            "thal",
            "target",
        ]
        self.categorical_columns = ["sex", "cp", "fbs", "restecg", "exang", "slope"]

    def _validate_dataframe(self, df: pd.DataFrame, dataset_name: str):
        if df.empty:
            raise ValueError(f"{dataset_name} dataframe is empty")

        missing_columns = [col for col in self.expected_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(
                f"{dataset_name} is missing required columns: {missing_columns}"
            )

        if "target" not in df.columns:
            raise ValueError(f"{dataset_name} is missing target column")

        for col in self.categorical_columns:
            if df[col].dropna().empty:
                raise ValueError(
                    f"{dataset_name} categorical column '{col}' has no valid values"
                )

        logging.info(f"{dataset_name} validation checks passed")

    def initiate_data_validation(self, train_path, test_path):
        try:
            logging.info("Entered the data validation method or component")

            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            self._validate_dataframe(train_df, "Train")
            self._validate_dataframe(test_df, "Test")

            validation_status = True
            os.makedirs(
                os.path.dirname(self.data_validation_config.validation_status_file_path),
                exist_ok=True,
            )
            with open(self.data_validation_config.validation_status_file_path, "w") as f:
                f.write(f"Validation status: {validation_status}")

            logging.info("Data validation completed successfully")
            return train_path, test_path

        except Exception as e:
            raise CustomException(e, sys)
