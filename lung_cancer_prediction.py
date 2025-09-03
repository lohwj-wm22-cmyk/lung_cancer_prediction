import streamlit as st
import pandas as pd
from joblib import load
from sklearn.preprocessing import MinMaxScaler

# Load the trained models
# lr_model = load('logreg_model.joblib')
rf_model = load('rf_model.joblib')
# svm_model = load('svm_model.joblib')
scaler = load('scaler.pkl') 
# scaler = MinMaxScaler()

# ---------------- Dataset Preview Page ----------------
def dataset_preview_page():
    st.title('📊 Dataset Preview')
    st.header('Lung Cancer Prediction Dataset')
    
    # Link to dataset (example placeholder - update with actual dataset link)
    dataset_link = 'https://www.kaggle.com/datasets/nancyalaswad90/lung-cancer'
    st.write(f'You can download the full dataset from [Kaggle]({dataset_link}).')
    
    # Load a sample dataset for preview
    df = pd.read_csv('lung_data.csv')  # Update this with your dataset file
    st.write('Here is a preview of the dataset:')
    st.dataframe(df.head(20))

# ---------------- Prediction Page ----------------
def prediction_page():
    st.title('🫁 Lung Cancer Prediction App')
    st.write('Fill in the patient details to predict the risk of lung cancer.')

    # Input fields for user data
    age = st.number_input('Age 🎂', min_value=0, max_value=120, value=50)
    gender = st.selectbox('Gender 👤', ['Male', 'Female'])
    smoking = st.selectbox('Do you smoke? 🚬', ['Yes', 'No'])
    yellow_fingers = st.selectbox('Yellow Fingers ✋', ['Yes', 'No'])
    anxiety = st.selectbox('Anxiety 😟', ['Yes', 'No'])
    peer_pressure = st.selectbox('Peer Pressure 👥', ['Yes', 'No'])
    chronic_disease = st.selectbox('Chronic Disease 🏥', ['Yes', 'No'])
    fatigue = st.selectbox('Fatigue 😴', ['Yes', 'No'])
    allergy = st.selectbox('Allergy 🤧', ['Yes', 'No'])
    wheezing = st.selectbox('Wheezing 😤', ['Yes', 'No'])
    alcohol_consumption = st.selectbox('Alcohol Consumption 🍺', ['Yes', 'No'])
    coughing = st.selectbox('Coughing 🤧', ['Yes', 'No'])
    shortness_of_breath = st.selectbox('Shortness of Breath 🫁', ['Yes', 'No'])
    swallowing_difficulty = st.selectbox('Swallowing Difficulty 😣', ['Yes', 'No'])
    chest_pain = st.selectbox('Chest Pain ❤️‍🩹', ['Yes', 'No'])

    # When user clicks Predict button
    if st.button('Predict 🔮'):
        # Create a dictionary for the input
        input_data = {
            'Age': [age],
            'Gender': [gender],
            'Smoking': [smoking],
            'Yellow_Fingers': [yellow_fingers],
            'Anxiety': [anxiety],
            'Peer_Pressure': [peer_pressure],
            'Chronic_Disease': [chronic_disease],
            'Fatigue': [fatigue],
            'Allergy': [allergy],
            'Wheezing': [wheezing],
            'Alcohol_Consumption': [alcohol_consumption],
            'Coughing': [coughing],
            'Shortness_of_Breath': [shortness_of_breath],
            'Swallowing_Difficulty': [swallowing_difficulty],
            'Chest_Pain': [chest_pain]
        }

        input_df = pd.DataFrame(input_data)

            # Define model columns
        model_columns = [
            'Age',
            'Gender_Female', 'Gender_Male',
            'Smoking_Yes', 'Smoking_No',
            'Yellow_Fingers_Yes', 'Yellow_Fingers_No',
            'Anxiety_Yes', 'Anxiety_No',
            'Peer_Pressure_Yes', 'Peer_Pressure_No',
            'Chronic_Disease_Yes', 'Chronic_Disease_No',
            'Fatigue_Yes', 'Fatigue_No',
            'Allergy_Yes', 'Allergy_No',
            'Wheezing_Yes', 'Wheezing_No',
            'Alcohol_Consumption_Yes', 'Alcohol_Consumption_No',
            'Coughing_Yes', 'Coughing_No',
            'Shortness_of_Breath_Yes', 'Shortness_of_Breath_No',
            'Swallowing_Difficulty_Yes', 'Swallowing_Difficulty_No',
            'Chest_Pain_Yes', 'Chest_Pain_No'
        ]

        # Create encoded dataframe
        encoded_input_df = pd.DataFrame(0, index=input_df.index, columns=model_columns)

        # Copy numeric column
        encoded_input_df['Age'] = input_df['Age']

        # Hardcode categorical mappings
        categorical_data = {
            'Gender': {'Male': 'Gender_Male', 'Female': 'Gender_Female'},
            'Smoking': {'Yes': 'Smoking_Yes', 'No': 'Smoking_No'},
            'Yellow_Fingers': {'Yes': 'Yellow_Fingers_Yes', 'No': 'Yellow_Fingers_No'},
            'Anxiety': {'Yes': 'Anxiety_Yes', 'No': 'Anxiety_No'},
            'Peer_Pressure': {'Yes': 'Peer_Pressure_Yes', 'No': 'Peer_Pressure_No'},
            'Chronic_Disease': {'Yes': 'Chronic_Disease_Yes', 'No': 'Chronic_Disease_No'},
            'Fatigue': {'Yes': 'Fatigue_Yes', 'No': 'Fatigue_No'},
            'Allergy': {'Yes': 'Allergy_Yes', 'No': 'Allergy_No'},
            'Wheezing': {'Yes': 'Wheezing_Yes', 'No': 'Wheezing_No'},
            'Alcohol_Consumption': {'Yes': 'Alcohol_Consumption_Yes', 'No': 'Alcohol_Consumption_No'},
            'Coughing': {'Yes': 'Coughing_Yes', 'No': 'Coughing_No'},
            'Shortness_of_Breath': {'Yes': 'Shortness_of_Breath_Yes', 'No': 'Shortness_of_Breath_No'},
            'Swallowing_Difficulty': {'Yes': 'Swallowing_Difficulty_Yes', 'No': 'Swallowing_Difficulty_No'},
            'Chest_Pain': {'Yes': 'Chest_Pain_Yes', 'No': 'Chest_Pain_No'}
        }

        # Encode categorical
        for col in categorical_data:
            for column in categorical_data[col].values():
                encoded_input_df[column] = 0
            value = input_df[col].iloc[0]
            encoded_input_df[categorical_data[col][value]] = 1

        # Ensure all columns are present
        encoded_input_df = encoded_input_df.reindex(columns=model_columns, fill_value=0)
       
        # Scale features
        if scaler:
            input_df_scaled = scaler.transform(input_df)

            # Predict using Random Forest model
            prediction = rf_model.predict(input_df_scaled)[0]

            # Display the prediction result
            st.success(f'🌟 Prediction: {"High Risk of Lung Cancer" if prediction == 1 else "Low Risk"}')
        else:
            st.error("⚠️ Scaler not loaded properly. Please check the scaler file.")

# ---------------- About Page ----------------
def about_page():
    st.title('📚 About the Project')
    st.header('Lung Cancer Prediction using Machine Learning Models')
    st.write("""
    This project aims to predict the likelihood of lung cancer based on patient health data 
    using a Random Forest model. The dataset includes risk factors such as smoking habits, 
    medical history, and respiratory symptoms.

    The goal is to assist healthcare professionals in identifying individuals 
    at high risk early, supporting preventive care and early diagnosis.
    """)

# ---------------- Main Function ----------------
def main():
    st.sidebar.title('🗂️ Navigation')
    menu_options = ['Prediction Page', 'Dataset Preview', 'About the Project']
    choice = st.sidebar.selectbox('Go to', menu_options)

    if choice == 'Prediction Page':
        prediction_page()
    elif choice == 'Dataset Preview':
        dataset_preview_page()
    elif choice == 'About the Project':
        about_page()

if __name__ == '__main__':
    main()









