import streamlit as st
import pickle
import numpy as np

# Load the model
model = pickle.load(open('model.pkl', 'rb'))

# App title
st.title("Breast Cancer Detection System")

# App description
st.write("Enter the feature values to predict whether the tumor is malignant (M) or benign (B).")

# Create input fields for features
st.header("Enter Feature Data")
features = [
    'radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean', 'smoothness_mean',
    'compactness_mean', 'concavity_mean', 'concave points_mean', 'symmetry_mean',
    'fractal_dimension_mean', 'radius_se', 'texture_se', 'perimeter_se', 'area_se',
    'smoothness_se', 'compactness_se', 'concavity_se', 'concave points_se',
    'symmetry_se', 'fractal_dimension_se', 'radius_worst', 'texture_worst',
    'perimeter_worst', 'area_worst', 'smoothness_worst', 'compactness_worst',
    'concavity_worst', 'concave points_worst', 'symmetry_worst',
    'fractal_dimension_worst'
]

# Create dictionary to store inputs
input_data = {}

# Create input fields for each feature
for feature in features:
    input_data[feature] = st.number_input(f"{feature}", value=0.0, format="%.6f")

# Prediction button
if st.button("Predict"):
    # Convert inputs to array
    input_array = np.array([input_data[feature] for feature in features]).reshape(1, -1)
    
    # Make prediction using the model
    prediction = model.predict(input_array)
    
    # Display result
    if prediction[0] == 1:
        st.error("Prediction: Malignant (M)")
    else:
        st.success("Prediction: Benign (B)")

# Additional notes
st.write("""
**Notes:**
- Ensure the input values are accurate.
- The model was trained using the Wisconsin Breast Cancer dataset.
- For best results, use values within the range of the original dataset.
""")