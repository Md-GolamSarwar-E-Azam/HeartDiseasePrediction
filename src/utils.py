import sys
import os

import dill
from sklearn.metrics import f1_score, make_scorer
from src.exception import CustomException
from sklearn.model_selection import GridSearchCV, StratifiedKFold

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)


def eveluate_models(X_train, y_train, models, param):
    try:
        report = {}
        best_params = {}
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        scoring = make_scorer(f1_score, zero_division=0)

        for model_name, model in models.items():
            gs = GridSearchCV(
                model,
                param[model_name],
                cv=cv,
                scoring=scoring,
                error_score="raise",
            )
            gs.fit(X_train, y_train)

            models[model_name] = gs.best_estimator_
            report[model_name] = gs.best_score_
            best_params[model_name] = gs.best_params_

        return report, best_params

    except Exception as e:
        raise CustomException(e, sys)


def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return dill.load(file_obj)


    except Exception as e:
        raise CustomException(e, sys)
        