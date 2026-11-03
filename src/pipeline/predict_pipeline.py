import sys 
from pathlib import Path

import pandas as pd 
from src.exception import CustomException 
from src.utils import load_object 


class PredictPipelline:
    def __init__(self):
        pass 

    def predict(self,features):
        try:
            project_root = Path(__file__).resolve().parents[2]
            model_path = project_root / "artifacts" / "model.pkl"
            preprocessor_path = project_root / "artifacts" / "preprocessor.pkl"
            model = load_object(file_path = model_path)
            preprocessor = load_object(file_path = preprocessor_path)
            data_scaled = preprocessor.transform(features)
            preds = model.predict(data_scaled)
            return preds

        except Exception as e:
            raise CustomException(e,sys)    


class CustomData:
    def __init__(self,
        age: int,
        sex: int,
        cp: int,
        trestbps:int,
        chol:int,
        fbs: int,
        restecg: int,
        thalach: int,
        exang: int,
        oldpeak: float,
        slope: int,
        ca: int,
        thal: int 
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


     

    def get_data_as_frame(self):
        try: 
            custom_data_input_dict = {
                "age": [self.age],
                "sex": [self.sex],
                "cp": [self.cp],
                "trestbps": [self.trestbps],
                "chol": [self.chol],
                "fbs": [self.fbs],
                "restecg": [self.restecg],
                "thalach": [self.thalach],
                "exang": [self.exang],
                "oldpeak": [self.oldpeak],
                "slope": [self.slope],
                "ca": [self.ca],
                "thal": [self.thal]
            }

            return pd.DataFrame(custom_data_input_dict)
        
        except Exception as e:
            raise CustomException(e, sys)
