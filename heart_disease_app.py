import streamlit as st
import numpy as np
import joblib

# Load the trained model and scaler
rf_model = joblib.load("random_forest_model.pkl")
sc = joblib.load("standard_scaler.pkl")

# Function to preprocess input data and make predictions
def predict(input_data):
    # Convert input data to numpy array
    input_array = np.array([[
        input_data["age"],
        input_data["sex"],
        input_data["cp"],
        input_data["trestbps"],
        input_data["chol"],
        input_data["fbs"],
        input_data["restecg"],
        input_data["thalach"],
        input_data["exang"],
        input_data["oldpeak"],
        input_data["slope"],
        input_data["ca"],
        input_data["thal"]
    ]])

    # Scale the input data
    input_array_scaled = sc.transform(input_array)

    # Make a prediction
    prediction = rf_model.predict(input_array_scaled)
    return prediction[0]

# Streamlit app
def main():
    st.title("Heart Disease Prediction App")
    st.write("Enter the patient's details to predict the risk of heart disease.")

    # Input fields for user data
    age = st.number_input("Age", min_value=0, max_value=120, value=58)
    sex = st.selectbox("Sex", options=[0, 1], format_func=lambda x: "Female" if x == 0 else "Male")
    cp = st.selectbox("Chest Pain Type (cp)", options=[0, 1, 2, 3])
    trestbps = st.number_input("Resting Blood Pressure (trestbps)", min_value=0, max_value=200, value=132)
    chol = st.number_input("Serum Cholesterol (chol)", min_value=0, max_value=600, value=224)
    fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl (fbs)", options=[0, 1])
    restecg = st.selectbox("Resting Electrocardiographic Results (restecg)", options=[0, 1, 2])
    thalach = st.number_input("Maximum Heart Rate Achieved (thalach)", min_value=0, max_value=220, value=173)
    exang = st.selectbox("Exercise Induced Angina (exang)", options=[0, 1])
    oldpeak = st.number_input("ST Depression Induced by Exercise (oldpeak)", min_value=0.0, max_value=10.0, value=3.2)
    slope = st.selectbox("Slope of the Peak Exercise ST Segment (slope)", options=[0, 1, 2])
    ca = st.number_input("Number of Major Vessels Colored by Fluoroscopy (ca)", min_value=0, max_value=4, value=2)
    thal = st.selectbox("Thalassemia (thal)", options=[1, 2, 3, 4, 5, 6, 7])

    # Create a dictionary of input data
    input_data = {
        "age": age,
        "sex": sex,
        "cp": cp,
        "trestbps": trestbps,
        "chol": chol,
        "fbs": fbs,
        "restecg": restecg,
        "thalach": thalach,
        "exang": exang,
        "oldpeak": oldpeak,
        "slope": slope,
        "ca": ca,
        "thal": thal
    }

    # Predict button
    if st.button("Predict"):
        # Make prediction
        prediction = predict(input_data)
        st.write(f"Predicted Target: {prediction}")

# Run the app
if __name__ == "__main__":
    main()