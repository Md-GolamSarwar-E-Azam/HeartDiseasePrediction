import sys
from pathlib import Path

import pandas as pd
from src.exception import CustomException
from src.utils import load_object


class PredictPipelline:
    def __init__(self):
        pass

    def predict(self, features):
        try:
            project_root = Path(__file__).resolve().parents[2]
            model_path = project_root / "artifacts" / "model.pkl"
            preprocessor_path = project_root / "artifacts" / "preprocessor.pkl"
            model = load_object(file_path=model_path)
            preprocessor = load_object(file_path=preprocessor_path)
            data_scaled = preprocessor.transform(features)
            preds = model.predict(data_scaled)
            return preds

        except Exception as e:
            raise CustomException(e, sys)


class CustomData:
    SEX_MAP = {"0": "female", "1": "male"}
    CP_MAP = {
        "0": "Typical angina",
        "1": "Atypical angina",
        "2": "Non-anginal pain",
        "3": "Asymptomatic",
    }
    FBS_MAP = {"0": "False", "1": "True"}
    RESTECG_MAP = {
        "0": "Nothing to note",
        "1": "ST-T Wave abnormality",
        "2": "Left ventricular hypertrophy",
    }
    EXANG_MAP = {"0": "no", "1": "yes"}
    SLOPE_MAP = {
        "0": "Upsloping",
        "1": "Flatsloping",
        "2": "Downsloping",
    }

    def __init__(
        self,
        age: int,
        sex: int,
        cp: int,
        trestbps: int,
        chol: int,
        fbs: int,
        restecg: int,
        thalach: int,
        exang: int,
        oldpeak: float,
        slope: int,
        ca: int,
        thal: int,
    ):
        self.age = age
        self.sex = sex
        self.cp = cp
        self.trestbps = trestbps
        self.chol = chol
        self.fbs = fbs
        self.restecg = restecg
        self.thalach = thalach
        self.exang = exang
        self.oldpeak = oldpeak
        self.slope = slope
        self.ca = ca
        self.thal = thal

    def _map_value(self, value, mapping, field_name):
        key = str(value)
        if key not in mapping:
            raise ValueError(
                f"Unsupported value '{value}' for field '{field_name}'"
            )
        return mapping[key]

    def get_data_as_frame(self):
        try:
            custom_data_input_dict = {
                "age": [float(self.age)],
                "sex": [self._map_value(self.sex, self.SEX_MAP, "sex")],
                "cp": [self._map_value(self.cp, self.CP_MAP, "cp")],
                "trestbps": [float(self.trestbps)],
                "chol": [float(self.chol)],
                "fbs": [self._map_value(self.fbs, self.FBS_MAP, "fbs")],
                "restecg": [self._map_value(self.restecg, self.RESTECG_MAP, "restecg")],
                "thalach": [float(self.thalach)],
                "exang": [self._map_value(self.exang, self.EXANG_MAP, "exang")],
                "oldpeak": [float(self.oldpeak)],
                "slope": [self._map_value(self.slope, self.SLOPE_MAP, "slope")],
                "ca": [float(self.ca)],
                "thal": [float(self.thal)],
            }

            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            raise CustomException(e, sys)
