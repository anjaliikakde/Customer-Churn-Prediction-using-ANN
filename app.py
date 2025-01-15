import streamlit as st
import tensorflow as tf
import pandas as pd
import pickle

# Load the trained model
model = tf.keras.models.load_model('model.h5')

# Load the encoders and scaler
with open('label_encoder_gender.pkl', 'rb') as file:
    label_encoder_gender = pickle.load(file)

with open('onehot_encoder_geo.pkl', 'rb') as file:
    onehot_encoder_geo = pickle.load(file)

with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

# Streamlit app
st.set_page_config(page_title="Customer Churn Prediction", page_icon="📊", layout="wide")

# Custom CSS for UI styling
st.markdown("""
    <style>
    body {
        background-color: #f8f9fa;
    }
    .main {
        background-color: #ffffff;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0px 4px 10px rgba(0, 0, 0, 0.1);
    }
    h1 {
        color: #1e88e5;
        text-align: center;
        font-family: 'Arial', sans-serif;
    }
    .stButton button {
        background-color: #1e88e5;
        color: white;
        border: none;
        border-radius: 5px;
        padding: 8px 16px;
        font-size: 16px;
        font-weight: bold;
    }
    .stButton button:hover {
        background-color: #1565c0;
    }
    </style>
    """, unsafe_allow_html=True)

st.title("📊 CUSTOMER CHURN PREDICTION")
st.markdown("Predict if a customer is likely to churn based on their details.")

# Sidebar for input parameters
with st.sidebar:
    st.header("Input Parameters")
    geography = st.selectbox('Geography', onehot_encoder_geo.categories_[0], help="Select the customer's region.")
    gender = st.selectbox('Gender', label_encoder_gender.classes_, help="Select the customer's gender.")
    age = st.slider('Age', 18, 92, 30, help="Select the customer's age.")
    balance = st.number_input('Balance', format="%.2f", help="Enter the customer's account balance.")
    credit_score = st.number_input('Credit Score', value=0, step=1, format="%d", help="Enter the customer's credit score.")
    estimated_salary = st.number_input('Estimated Salary', format="%.2f", help="Enter the customer's estimated salary.")
    tenure = st.slider('Tenure', 0, 10, 5, help="Select the tenure of the customer.")
    num_of_products = st.slider('Number of Products', 1, 4, 1, help="Select the number of products owned by the customer.")
    has_cr_card = st.selectbox('Has Credit Card', [0, 1], format_func=lambda x: "Yes" if x else "No", help="Does the customer have a credit card?")
    is_active_member = st.selectbox('Is Active Member', [0, 1], format_func=lambda x: "Yes" if x else "No", help="Is the customer an active member?")

# Prepare the input data
input_data = pd.DataFrame({
    'CreditScore': [credit_score],
    'Gender': [label_encoder_gender.transform([gender])[0]],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [has_cr_card],
    'IsActiveMember': [is_active_member],
    'EstimatedSalary': [estimated_salary]
})

# One-hot encode 'Geography'
geo_encoded = onehot_encoder_geo.transform([[geography]]).toarray()
geo_encoded_df = pd.DataFrame(geo_encoded, columns=onehot_encoder_geo.get_feature_names_out(['Geography']))

# Combine one-hot encoded columns with input data
input_data = pd.concat([input_data.reset_index(drop=True), geo_encoded_df], axis=1)

# Scale the input data
input_data_scaled = scaler.transform(input_data)

# Main area for prediction results
st.subheader("Prediction Results")

if st.button("Predict Churn"):
    prediction = model.predict(input_data_scaled)
    prediction_proba = prediction[0][0]

    st.markdown(f"### Churn Probability: **{prediction_proba:.2f}**")

    if prediction_proba > 0.5:
        st.error("🔴 The customer is **likely** to churn.")
    else:
        st.success("🟢 The customer is **not likely** to churn.")
