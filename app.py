import streamlit as st
import numpy as np
import pandas as pd
import joblib

#load the model
model= joblib.load("model/model.pkl")
scaler = joblib.load("model/scaler.pkl")
feature_cols = joblib.load("model/feature_cols.pkl")
st.title("Iris Prediction model")
#input features
SepalLength = st.slider("SepalLength(cm)",4.0,5.0,8.0)
SepalWidthCm=st.slider("SepalWidth(cm)", 2.0,5.0,3.0)	
PetalLengthCm= st.slider("PetalLength(cm)",1.0,7.0,4.0 )	
PetalWidthCm= st.slider("PetalWidth(cm)", 0.1, 2.5, 1.0)
# input prediction

input_data= pd.DataFrame(
    [[SepalLength,SepalWidthCm, PetalLengthCm, PetalWidthCm]],
    columns=feature_cols)

new_scaled= scaler.transform(input_data)
prediction= model.predict(new_scaled)

predicted_class = {0 : "Iris-setosa", 1:  "Iris-versicolor", 2: "Iris-virginica"}[0]



label_map = {
    0: "Iris-setosa",
    1: "Iris-versicolor",
    2: "Iris-virginica"
}
class_id = int(prediction[0])
flower_name = label_map[class_id]
st.write(f"Predicted Iris flower : [{class_id}] → {flower_name}")
