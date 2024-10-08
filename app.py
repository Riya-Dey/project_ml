from flask import Flask,request,render_template
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from src.pipeline.predict_pipeline import CustomData,PredictPipeline

application=Flask(__name__)

app=application

## Route for a home page

@app.route('/')
def index():
    return render_template('index.html') 

@app.route('/predictdata',methods=['GET','POST'])
def predict_datapoint():
    if request.method=='GET':
        return render_template('home.html')
    else:
        data=CustomData(
            PM2_5=request.form.get('PM2_5'),
            NO2=request.form.get('NO2'),
            CO=request.form.get('CO'),
            SO2=request.form.get('SO2'),
            O3=request.form.get('O3')
        )
        pred_df=data.get_data_as_data_frame()
        print(pred_df)
        print("Before Prediction")

        predict_pipeline=PredictPipeline()
        print("Mid Prediction")
        results=predict_pipeline.predict(pred_df)
        print("after Prediction")
        rounded_results = [round(result, 2) for result in results]
        return render_template('home.html',results=rounded_results[0])
    

if __name__=="__main__":
    app.run(host="0.0.0.0", debug=True)        