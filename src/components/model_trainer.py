import os
import sys
from dataclasses import dataclass

from catboost import CatBoostClassifier
from sklearn.ensemble import (
    AdaBoostClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from src.exception import CustomException
from src.logger import logging
from src.utils import eveluate_models, save_object


@dataclass
class ModelTrainerConfig:
    trained_model_file_path: str = os.path.join("artifacts", "model.pkl")


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Splitting training and test input data")
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1],
            )
            models = {
                "Random Forest": RandomForestClassifier(random_state=42),
                "Decision Tree": DecisionTreeClassifier(random_state=42),
                "Gradient Boosting": GradientBoostingClassifier(random_state=42),
                "Logistic Regression": LogisticRegression(max_iter=1000),
                "K-Neighbors Classifier": KNeighborsClassifier(),
                "XGBClassifier": XGBClassifier(
                    eval_metric="logloss",
                    random_state=42,
                ),
                "CatBoosting Classifier": CatBoostClassifier(
                    verbose=False,
                    random_seed=42,
                ),
                "AdaBoost Classifier": AdaBoostClassifier(random_state=42),
            }
            params = {
                "Decision Tree": {
                    "criterion": ["gini", "entropy", "log_loss"],
                    "splitter": ["best", "random"],
                    "max_features": ["sqrt", "log2"],
                },
                "Random Forest": {
                    "n_estimators": [50, 100, 200],
                    "max_features": ["sqrt", "log2"],
                },
                "Gradient Boosting": {
                    "learning_rate": [0.05, 0.1],
                    "n_estimators": [50, 100],
                    "max_depth": [2, 3],
                },
                "Logistic Regression": {
                    "C": [0.1, 1.0, 10.0],
                    "class_weight": [None, "balanced"],
                },
                "K-Neighbors Classifier": {
                    "n_neighbors": [3, 5, 7, 9],
                    "weights": ["uniform", "distance"],
                },
                "XGBClassifier": {
                    "learning_rate": [0.05, 0.1],
                    "n_estimators": [50, 100],
                    "max_depth": [3, 5],
                },
                "CatBoosting Classifier": {
                    "depth": [4, 6, 8],
                    "learning_rate": [0.03, 0.1],
                    "iterations": [50, 100],
                },
                "AdaBoost Classifier": {
                    "learning_rate": [0.05, 0.1, 0.5],
                    "n_estimators": [50, 100],
                },
            }

            model_report: dict = eveluate_models(
                X_train=X_train,
                y_train=y_train,
                models=models,
                param=params,
            )

            best_model_name = max(model_report, key=model_report.get)
            best_model_score = model_report[best_model_name]
            best_model = models[best_model_name]

            logging.info(
                f"Best model selected by cross-validation F1 is "
                f"{best_model_name} with score: {best_model_score}"
            )

            predicted = best_model.predict(X_test)
            test_f1 = f1_score(y_test, predicted, zero_division=0)
            test_precision = precision_score(y_test, predicted, zero_division=0)
            test_recall = recall_score(y_test, predicted, zero_division=0)

            logging.info(
                f"Test metrics for {best_model_name}: "
                f"F1={test_f1}, precision={test_precision}, recall={test_recall}"
            )

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model,
            )

            return test_f1, best_model_name

        except Exception as e:
            raise CustomException(e, sys)