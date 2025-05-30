import streamlit as st
import pandas as pd
import joblib


model = joblib.load("iris_knn_model.pkl")

st.title("🌸 Iris Flower Classifier ")

sepal_length = st.slider("Sepal Length", 4.0, 8.0, 5.1)
sepal_width = st.slider("Sepal Width", 2.0, 4.5, 3.5)
petal_length = st.slider("Petal Length", 1.0, 7.0, 1.4)
petal_width = st.slider("Petal Width", 0.1, 2.5, 0.2)

if st.button("Predict"):
    input_df = pd.DataFrame([[sepal_length, sepal_width, petal_length, petal_width]],
                             columns=["sepal.length", "sepal.width", "petal.length", "petal.width"])
    prediction = model.predict(input_df)[0]
    st.success(f"🌼 Predicted Variety: **{prediction}**")
