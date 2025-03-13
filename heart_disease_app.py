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

def get_risk_level(prediction):
    if prediction == 0:
        return "No Risk"
    elif prediction == 1:
        return "Low Risk"
    elif prediction == 2:
        return "Medium Risk"
    elif prediction == 3:
        return "High Risk"
    elif prediction == 4:
        return "Highest Risk"
    else:
        return "Unknown Risk"

def get_medication_recommendations(risk_level):
    recommendations = {
        "No Risk": [
            "Maintain a healthy lifestyle.",
            "Regular exercise and a balanced diet.",
            "No specific medication required."
        ],
        "Low Risk": [
            "Lifestyle changes: Exercise, healthy diet, and stress management.",
            "Consider low-dose aspirin if recommended by a doctor.",
            "Monitor blood pressure and cholesterol regularly."
        ],
        "Medium Risk": [
            "Medications: Statins to control cholesterol.",
            "Beta-blockers or ACE inhibitors for blood pressure management.",
            "Lifestyle changes: Quit smoking, reduce alcohol intake, and exercise regularly."
        ],
        "High Risk": [
            "Medications: Statins, beta-blockers, ACE inhibitors, and antiplatelet drugs.",
            "Regular monitoring of heart health.",
            "Consider consulting a cardiologist for advanced treatment options."
        ],
        "Highest Risk": [
            "Immediate medical attention required.",
            "Medications: High-dose statins, beta-blockers, ACE inhibitors, and antiplatelet drugs.",
            "Possible surgical interventions: Angioplasty or bypass surgery."
        ]
    }
    return recommendations.get(risk_level, ["No recommendations available."])
    
# Predefined sets of inputs
predefined_sets = {
    "Set 1": {
        "age": 58,
        "sex": 1,
        "cp": 3,
        "trestbps": 132,
        "chol": 224,
        "fbs": 0,
        "restecg": 2,
        "thalach": 173,
        "exang": 0,
        "oldpeak": 3.2,
        "slope": 1,
        "ca": 2,
        "thal": 7
    },
    "Set 2": {
        "age": 45,
        "sex": 0,
        "cp": 2,
        "trestbps": 130,
        "chol": 234,
        "fbs": 0,
        "restecg": 2,
        "thalach": 175,
        "exang": 0,
        "oldpeak": 0.6,
        "slope": 2,
        "ca": 0,
        "thal": 3
    },
    "Set 3": {
        "age": 61,
        "sex": 1,
        "cp": 4,
        "trestbps": 138,
        "chol": 166,
        "fbs": 0,
        "restecg": 2,
        "thalach": 125,
        "exang": 1,
        "oldpeak": 3.6,
        "slope": 2,
        "ca": 1,
        "thal": 3
    },
    "Set 4": {
        "age": 59,
        "sex": 1,
        "cp": 4,
        "trestbps": 170,
        "chol": 326,
        "fbs": 0,
        "restecg": 2,
        "thalach": 140,
        "exang": 1,
        "oldpeak": 3.4,
        "slope": 3,
        "ca": 0,
        "thal": 7
    },
    "Set 5": {
        "age": 58,
        "sex": 1,
        "cp": 2,
        "trestbps": 120,
        "chol": 284,
        "fbs": 0,
        "restecg": 2,
        "thalach": 165,
        "exang": 0,
        "oldpeak": 1.8,
        "slope": 2,
        "ca": 0,
        "thal": 3
    },
    "Set 6": {
        "age": 61,
        "sex": 1,
        "cp": 4,
        "trestbps": 138,
        "chol": 166,
        "fbs": 0,
        "restecg": 2,
        "thalach": 125,
        "exang": 1,
        "oldpeak": 3.6,
        "slope": 2,
        "ca": 1,
        "thal": 3
    }
}

# Streamlit app
def main():
    st.title("Heart Disease Prediction App")
    st.write("Enter the patient's details to predict the risk of heart disease.")

    # Dropdown menu for predefined sets
    selected_set = st.selectbox("Select a predefined set of inputs", options=list(predefined_sets.keys()))

    # Load the selected predefined set
    predefined_data = predefined_sets[selected_set]

    # Input fields for user data
    age = st.number_input("Age", min_value=0, max_value=120, value=predefined_data["age"])
    sex = st.selectbox("Sex", options=[0, 1], format_func=lambda x: "Female" if x == 0 else "Male", index=predefined_data["sex"])
    cp = st.selectbox("Chest Pain Type (cp)", options=[0,1, 2, 3], index=predefined_data["cp"]-1)
    trestbps = st.number_input("Resting Blood Pressure (trestbps)", min_value=0, max_value=200, value=predefined_data["trestbps"])
    chol = st.number_input("Serum Cholesterol (chol)", min_value=0, max_value=600, value=predefined_data["chol"])
    fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl (fbs)", options=[0, 1], index=predefined_data["fbs"])
    restecg = st.selectbox("Resting Electrocardiographic Results (restecg)", options=[0, 1, 2], index=predefined_data["restecg"])
    thalach = st.number_input("Maximum Heart Rate Achieved (thalach)", min_value=0, max_value=220, value=predefined_data["thalach"])
    exang = st.selectbox("Exercise Induced Angina (exang)", options=[0, 1], index=predefined_data["exang"])
    oldpeak = st.number_input("ST Depression Induced by Exercise (oldpeak)", min_value=0.0, max_value=10.0, value=predefined_data["oldpeak"])
    slope = st.selectbox("Slope of the Peak Exercise ST Segment (slope)", options=[0, 1, 2], index=predefined_data["slope"]-1)
    ca = st.number_input("Number of Major Vessels Colored by Fluoroscopy (ca)", min_value=0, max_value=4, value=predefined_data["ca"])
    thal = st.selectbox("Thalassemia (thal)", options=[1, 2, 3, 4, 5, 6, 7], index=predefined_data["thal"] - 1)

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
        risk_level = get_risk_level(prediction)
        st.write(f"Predicted Target: {prediction}")
        st.write(f"Risk Level: {risk_level}")

        # Display medication recommendations
        st.subheader("Medication Recommendations:")
        recommendations = get_medication_recommendations(risk_level)
        for recommendation in recommendations:
            st.write(f"- {recommendation}")

# Run the app
if __name__ == "__main__":
    main()
