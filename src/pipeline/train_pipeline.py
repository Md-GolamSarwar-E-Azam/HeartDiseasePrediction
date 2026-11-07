import sys

from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.data_validation import DataValidation
from src.components.model_trainer import ModelTrainer
from src.exception import CustomException
from src.logger import logging


class TrainPipeline:
    def initiate_train_pipeline(self):
        try:
            logging.info("Starting training pipeline")

            logging.info("[1/4] Data ingestion started")
            data_ingestion = DataIngestion()
            train_data_path, test_data_path = data_ingestion.initiate_data_ingestion()
            logging.info("[1/4] Data ingestion completed")

            logging.info("[2/4] Data validation started")
            data_validation = DataValidation()
            train_data_path, test_data_path = data_validation.initiate_data_validation(
                train_data_path, test_data_path
            )
            logging.info("[2/4] Data validation completed")

            logging.info("[3/4] Data transformation started")
            data_transformation = DataTransformation()
            train_arr, test_arr, _ = data_transformation.initiate_data_transformation(
                train_data_path, test_data_path
            )
            logging.info("[3/4] Data transformation completed")

            logging.info("[4/4] Model training started")
            model_trainer = ModelTrainer()
            test_f1, best_model_name = model_trainer.initiate_model_trainer(
                train_arr, test_arr
            )
            logging.info("[4/4] Model training completed")

            logging.info(
                f"Training pipeline completed. Best model: {best_model_name}, "
                f"test F1: {test_f1}"
            )
            return test_f1, best_model_name

        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    logging.info("Launching train pipeline from terminal")
    train_pipeline = TrainPipeline()
    result = train_pipeline.initiate_train_pipeline()
    logging.info(f"Final result: {result}")
    print(result)
