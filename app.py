import streamlit as st
import pickle
import pandas as pd
import numpy as np
from sklearn.datasets import load_iris

# Load the trained model
with open("iris_model.pkl", "rb") as f:
    model = pickle.load(f)

# Load dataset metadata
iris = load_iris()

# Streamlit UI
st.title("🌺 Iris Flower Species Prediction")
st.write("Enter the flower's measurements below to predict its species.")

# Input fields
sepal_length = st.number_input("Sepal Length (cm)", min_value=4.0, max_value=8.0, step=0.1)
sepal_width = st.number_input("Sepal Width (cm)", min_value=2.0, max_value=5.0, step=0.1)
petal_length = st.number_input("Petal Length (cm)", min_value=1.0, max_value=7.0, step=0.1)
petal_width = st.number_input("Petal Width (cm)", min_value=0.1, max_value=3.0, step=0.1)

# Predict button
if st.button("Predict"):
    # Create DataFrame with correct feature names
    input_features = pd.DataFrame([[sepal_length, sepal_width, petal_length, petal_width]],
                                  columns=iris.feature_names)
    
    # Predict the species
    prediction = model.predict(input_features)[0]
    species = iris.target_names[prediction]

    # Display the result
    st.success(f"🌼 Predicted Species: **{species.capitalize()}**")