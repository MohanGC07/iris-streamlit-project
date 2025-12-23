# app.py
import streamlit as st
import joblib
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Fixes black screen on Windows
import matplotlib.pyplot as plt

# --------------------------
# Title & description
# --------------------------
st.title("🌸 Iris Flower Prediction App")
st.write("Enter the measurements of the Iris flower below to predict its species.")

# --------------------------
# Load trained model safely
# --------------------------
try:
    model = joblib.load("iris_model.pkl")
    st.success("Model loaded successfully ✅")
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()  # Stop execution if model is not available

# --------------------------
# User input section
# --------------------------
sepal_length = st.number_input("Sepal Length (cm)", 0.0, 10.0, 5.1)
sepal_width  = st.number_input("Sepal Width (cm)",  0.0, 10.0, 3.5)
petal_length = st.number_input("Petal Length (cm)", 0.0, 10.0, 1.4)
petal_width  = st.number_input("Petal Width (cm)",  0.0, 10.0, 0.2)

# Prepare input for prediction
input_features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])

# --------------------------
# Prediction logic
# --------------------------
if st.button("Predict"):
    try:
        prediction = model.predict(input_features)
        species = ["Setosa", "Versicolor", "Virginica"]
        predicted_species = species[prediction[0]]
        
        st.success(f"Predicted Species: **{predicted_species}**")

        # --------------------------
        # Feature visualization
        # --------------------------
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.bar(
            ['Sepal L', 'Sepal W', 'Petal L', 'Petal W'],
            input_features[0],
            color=['#FF9999','#66B2FF','#99FF99','#FFCC99']
        )
        ax.set_ylim(0, max(input_features[0])+1)
        ax.set_title("Feature Values of Input Flower")
        st.pyplot(fig)

    except Exception as e:
        st.error(f"Error during prediction: {e}")
#Deploy 