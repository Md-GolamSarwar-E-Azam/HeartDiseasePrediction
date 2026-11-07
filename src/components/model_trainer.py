import os
import sys
from dataclasses import dataclass

import mlflow
import dagshub
dagshub.init(repo_owner='Md-GolamSarwar-E-Azam', repo_name='HeartDiseasePrediction', mlflow=True)

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
    preprocessor_obj_file_path: str = os.path.join("artifacts", "preprocessor.pkl")


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def _log_params_safe(self, params: dict):
        for key, value in params.items():
            mlflow.log_param(key, "None" if value is None else value)

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

            model_report, best_params = eveluate_models(
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

            test_metrics = {}
            for model_name, model in models.items():
                model_predicted = model.predict(X_test)
                test_metrics[model_name] = {
                    "test_f1": f1_score(y_test, model_predicted, zero_division=0),
                    "test_precision": precision_score(
                        y_test, model_predicted, zero_division=0
                    ),
                    "test_recall": recall_score(
                        y_test, model_predicted, zero_division=0
                    ),
                }

            test_f1_winner = max(
                test_metrics, key=lambda name: test_metrics[name]["test_f1"]
            )
            cv_test_agreement = best_model_name == test_f1_winner
            selection_rank = {
                name: rank
                for rank, name in enumerate(
                    sorted(model_report, key=model_report.get, reverse=True),
                    start=1,
                )
            }
            test_f1_rank = {
                name: rank
                for rank, name in enumerate(
                    sorted(
                        test_metrics,
                        key=lambda n: test_metrics[n]["test_f1"],
                        reverse=True,
                    ),
                    start=1,
                )
            }

            if cv_test_agreement:
                logging.info(
                    f"CV winner and test_f1 winner agree: {best_model_name}"
                )
            else:
                logging.info(
                    f"CV winner={best_model_name}, test_f1 winner={test_f1_winner} "
                    f"— deploying CV winner"
                )

            tracking_uri = mlflow.get_tracking_uri()
            logging.info(f"MLflow tracking URI: {tracking_uri}")
            mlflow.set_experiment("heart_disease_prediction")

            with mlflow.start_run(run_name="model_comparison"):
                mlflow.log_param("train_rows", X_train.shape[0])
                mlflow.log_param("test_rows", X_test.shape[0])
                mlflow.log_param("n_features", X_train.shape[1])
                mlflow.log_param("random_state", 42)
                mlflow.log_param("selection_metric", "cv_f1")
                mlflow.log_param("selection_rule", "highest_cv_f1")
                mlflow.log_param("best_model", best_model_name)
                mlflow.log_param("cv_winner", best_model_name)
                mlflow.log_param("test_f1_winner", test_f1_winner)
                mlflow.set_tag(
                    "cv_test_agreement",
                    "true" if cv_test_agreement else "false",
                )
                mlflow.log_metric("cv_f1", float(best_model_score))
                mlflow.log_metric("test_f1", float(test_f1))
                mlflow.log_metric("test_precision", float(test_precision))
                mlflow.log_metric("test_recall", float(test_recall))
                mlflow.log_artifact(self.model_trainer_config.trained_model_file_path)
                mlflow.log_artifact(
                    self.model_trainer_config.preprocessor_obj_file_path
                )

                for model_name in models:
                    metrics = test_metrics[model_name]
                    with mlflow.start_run(run_name=model_name, nested=True):
                        mlflow.log_param("model_name", model_name)
                        mlflow.log_param("cv_folds", 3)
                        mlflow.log_param("scoring", "f1")
                        mlflow.log_param("selection_metric", "cv_f1")
                        self._log_params_safe(best_params[model_name])
                        mlflow.log_metric("cv_f1", float(model_report[model_name]))
                        mlflow.log_metric("test_f1", float(metrics["test_f1"]))
                        mlflow.log_metric(
                            "test_precision", float(metrics["test_precision"])
                        )
                        mlflow.log_metric(
                            "test_recall", float(metrics["test_recall"])
                        )
                        mlflow.set_tag(
                            "is_best",
                            "true" if model_name == best_model_name else "false",
                        )
                        mlflow.set_tag(
                            "selection_rank", str(selection_rank[model_name])
                        )
                        mlflow.set_tag(
                            "test_f1_rank", str(test_f1_rank[model_name])
                        )

            logging.info(
                "MLflow logging completed for all models under model_comparison run"
            )

            return test_f1, best_model_name

        except Exception as e:
            raise CustomException(e, sys)
