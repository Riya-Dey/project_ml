import os
import sys
import pandas as pd
from src.exception import CustomException
from src.utils import load_object


class PredictPipeline:
    def __init__(self):
        pass

    def predict(self,features):
        try:
            model_path=os.path.join("artifacts","model.pkl")
            preprocessor_path=os.path.join('artifacts','preprocessor.pkl')
            print("Before Loading")
            model=load_object(file_path=model_path)
            preprocessor=load_object(file_path=preprocessor_path)
            print("After Loading")
            data_scaled=preprocessor.transform(features)
            preds=model.predict(data_scaled)
            return preds
        
        except Exception as e:
            raise CustomException(e,sys)

class CustomData:
    def __init__(  self, 
        PM2_5: float,
        NO2: float,
        CO: float,
        SO2: float,
        O3: float
        ):

        self.PM2_5 = PM2_5

        self.NO2 = NO2

        self.CO = CO

        self.SO2 = SO2

        self.O3 = O3

    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
                "PM2_5": [self.PM2_5],
                "NO2": [self.NO2],
                "CO": [self.CO],
                "SO2": [self.SO2],
                "O3": [self.O3]
            }

            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            raise CustomException(e, sys)