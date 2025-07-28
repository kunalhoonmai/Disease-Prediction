#importing libraries

import pickle
import numpy as np
import streamlit as st
from streamlit_option_menu import option_menu

# Set page configuration
st.set_page_config(page_title="Web-Based Multi-Disease Diagnosis System", layout="wide", page_icon="🏥")

#loading the models
diabetes_model=pickle.load(open("model/diabetes_model.pkl", "rb"))
heart_model=pickle.load(open("model/heart_model.pkl", "rb"))
liver_model=pickle.load(open("model/liver_model.pkl", "rb"))
breast_cancer_model=pickle.load(open("model/breast_cancer.pickle", "rb"))
chronic_kidney_model=pickle.load(open("model/kidney_model.pkl", "rb"))
lung_cancer_model=pickle.load(open("model/lung_cancer.pkl", "rb"))
parkinsons_model=pickle.load(open("model/parkinsons_model.pkl", "rb"))

#sidebar for navigation
with st.sidebar:
    selected=option_menu('Web-Based Multi-Disease Diagnosis System',
                         
                         ['Diabetes Prediction',
                          'Heart Disease Prediction',
                          'Liver Disease Prediction',
                          'Breast Cancer Disease Prediction',
                          'Chronic Kidney Disease Prediction',
                          'Lung Cancer Prediction',
                          'Parkinsons Disease Prediction'],

                          menu_icon='hospital-fill',

                          icons=[
                            'droplet-half',                 # Diabetes
                            'heart-pulse',                  # Heart
                            'shield-exclamation',           # Liver
                            'gender-female',                # Breast Cancer
                            'capsule',                      # Kidney
                            'lungs',                        # Lung Cancer
                            'activity'                      # Parkinson’s
                        ],

                          default_index=0)
    
#diabetes prediction page

if(selected=='Diabetes Prediction'):
    st.title('Diabetes Prediction') #page title

    # column input :
    #     gender
    #     age
    #     hypertension
    #     heart_disease
    #     smoking_history
    #     bmi(body mass index)
    #     HbA1c_level(hemoglobin)
    #     blood_glucose_level
    
    # getting the input data from the user
    col1, col2 = st.columns(2)
    
    with col1:
        gender = st.selectbox('Gender', ['Others (0)', 'Male (1)', 'Female (2)'])
        gender = ['Male (1)', 'Female (2)', 'Others (0)'].index(gender)

    with col2:
        age = st.text_input('Enter Your Age', placeholder="Age")

    with col1:
        hypertension = st.selectbox('Do You Have High Blood Pressure?', ['No (0)', 'Yes (1)'])
        hypertension = ['No (0)', 'Yes (1)'].index(hypertension)
    with col2:
        heart_disease = st.selectbox("Do You Have Heart Disease?", ['No (0)', 'Yes (1)'])
        heart_disease = ['No (0)', 'Yes (1)'].index(heart_disease)

    
    smoking_history = st.selectbox(
        "Do You Have Any Smoking History?",
        ['ever (0)', 'not current (1)', 'current(2)', 'former (3)', 'No Info (4)', 'never (5)']
    )
    smoking_history = ['ever (0)', 'not current (1)', 'current(2)', 'former (3)', 'No Info (4)', 'never (5)'].index(smoking_history)

    with col1:
        bmi = st.text_input('Enter Your Body Mass Index', placeholder="BMI")

    with col2:
       HbA1c_level = st.text_input('Enter Your Hemoglobin Level', placeholder="Hemoglobin")

    with col1:
        blood_glucose_level = st.text_input('Enter Your Blood Glucose Level', placeholder="Blood Glucose")


    # code for Prediction
    diab_diagnosis = ''

    # creating a button for Prediction
    if st.button('Diabetes Test Result'):

        user_input = [gender, age, hypertension, heart_disease, smoking_history, bmi, HbA1c_level, blood_glucose_level]

        user_input = [float(x) for x in user_input]

        diab_prediction = diabetes_model.predict([user_input])

        if diab_prediction[0] == 1:
            diab_diagnosis = 'The person is diabetic'

        else:
            diab_diagnosis = 'The person is not diabetic'

    st.success(diab_diagnosis)

#heart prediction page

if(selected=='Heart Disease Prediction'):
    st.title('Heart Disease Prediction') #page title
    

    Age = st.text_input('Enter Your Age', placeholder="Age")

    Sex = st.selectbox('Sex', ['Male (0)', 'Female (1)'])
    Sex = 1 if Sex == 'Female' else 0

        # Chest pain type:
        #     0: Typical angina
        #     1: Atypical angina
        #     2: Non-anginal pain
        #     3: Asymptomatic

    ChestPainType = st.selectbox('Chest Pain Type', ['Typical Angina (0)', 'Atypical Angina (1)', 'Non-anginal Pain (2)', 'Asymptomatic (3)'])
    ChestPainType = ['Typical Angina (0)', 'Atypical Angina (1)', 'Non-anginal Pain (2)', 'Asymptomatic (3)'].index(ChestPainType)

    RestingBP = st.text_input('Your Resting Blood Pressure in mm Hg', placeholder="BP")

    Cholesterol = st.text_input('Enter Your Serum Cholestoral in mg/dl', placeholder="Cholestoral")

    FastingBS = st.text_input("Blood Sugar Level is above 120 mg/dl", placeholder="Sugar Level")

        # Resting electrocardiographic results:
        #     0: Normal
        #     1: Having ST-T wave abnormality
        #     2: Showing probable or definite left ventricular hypertrophy

    RestingECG = st.selectbox('Resting ECG Results', ['Normal (2)', 'ST-T Wave Abnormality (0)', 'Left Ventricular Hypertrophy (1)'])
    RestingECG = ['Normal (2)', 'ST-T Wave Abnormality (0)', 'Left Ventricular Hypertrophy (1)'].index(RestingECG)

    MaxHR = st.text_input('Maximum Heart Rate Achieved During a Stress Test', placeholder="Max Heart Rate")

    ExerciseAngina = st.selectbox('Exercise-induced Angina', ['No (0)', 'Yes (1)'])
    ExerciseAngina = 0 if ExerciseAngina == 'No (0)' else 1

    Oldpeak = st.text_input("Enter Your Oldpeak ST Depression Induced By Exercise", placeholder="Oldpeak")

        # Slope of the peak exercise ST segment:
        #     0: Upsloping
        #     1: Flat
        #     2: Downsloping

    ST_Slope = st.selectbox('ST Slope', ['Upsloping (1)', 'Flat (2)', 'Downsloping (0)'])
    ST_Slope = ['Upsloping (1)', 'Flat (2)', 'Downsloping (0)'].index(ST_Slope)

    # code for Prediction
    heart_diagnosis = ''

    # creating a button for Prediction

    if st.button('Heart Disease Test Result'):

        user_input = [Age, Sex, ChestPainType, RestingBP, Cholesterol, FastingBS, RestingECG, MaxHR, ExerciseAngina, Oldpeak, ST_Slope]

        user_input = [float(x) for x in user_input]

        heart_prediction = heart_model.predict([user_input])

        if heart_prediction[0] == 1:
            heart_diagnosis = 'The person is having heart disease'
        else:
            heart_diagnosis = 'The person does not have any heart disease'

    st.success(heart_diagnosis)

#liver prediction page

if(selected=='Liver Disease Prediction'):
    st.title('Liver Disease Prediction') #page title

    col1, col2 = st.columns(2) # getting the input data from the user

    with col1:
        Gender = st.selectbox('Gender', ['Male (0)', 'Female (1)'])
        Gender = 1 if Gender == 'Female' else 0

    with col2:
        Age = st.text_input('Age', placeholder="Age")

    with col1:
        BMI = st.text_input('Body Mass Index', placeholder="BMI")

    with col2:
        AlcoholConsumption = st.text_input('Alcohol Consumption', placeholder="Alcohol Consumption")

    with col1:
        Smoking = st.selectbox("Do You Smoke?", ['No (0)', 'Yes (1)'])
        Smoking = 0 if Smoking == 'No' else 1

    with col2:
        GeneticRisk = st.selectbox("Genetic Risk", ['Low (0)', 'Medium (1)', 'High (2)'])
        GeneticRisk = ['Low (0)', 'Medium (1)', 'High (2)'].index(GeneticRisk)

    with col1:
        PhysicalActivity = st.text_input('Physical Activity', placeholder="Physical Activity")

    with col2:
        Diabetes = st.selectbox("Diabetes", ['No (0)', 'Yes (1)'])
        Diabetes = ['No (0)', 'Yes (1)'].index(Diabetes)

    with col1:
        Hypertension = st.selectbox("Hypertension", ['No (0)', 'Yes (1)'])
        Hypertension = ['No (0)', 'Yes (1)'].index(Hypertension)

    with col2:
        LiverFunctionTest = st.text_input('Liver Function Test', placeholder="Liver Function Test")

    # code for Prediction
    liver_diagnosis = ''

    # creating a button for Prediction    
    if st.button("Liver's Test Result"):

        user_input = [Gender, Age, BMI, AlcoholConsumption, Smoking, GeneticRisk, PhysicalActivity, Diabetes, Hypertension, LiverFunctionTest]

        user_input = [float(x) for x in user_input]

        liver_prediction = liver_model.predict([user_input])

        if liver_prediction[0] == 1:
            liver_diagnosis = "The person has Liver's disease"
        else:
            liver_diagnosis = "The person does not have Liver's disease"

    st.success(liver_diagnosis)

#Breast Cancer Prediction Page
if(selected=='Breast Cancer Disease Prediction'):
    st.title('Breast Cancer Prediction')

    # Input columns for user data
    col1, col2 = st.columns(2)

    with col1:
        mean_radius = st.number_input('Enter Mean Radius', placeholder="Mean Radius")

    with col2:
        mean_texture = st.number_input('Enter Mean Texture', placeholder="Mean Texture")

    with col1:
        mean_perimeter = st.number_input('Enter Mean Perimeter', placeholder="Mean Perimeter")

    with col2:
        mean_area = st.number_input('Enter Mean Area', placeholder="Mean Area")

    with col1:
        mean_smoothness = st.number_input('Enter Mean Smoothness', placeholder="Mean Smoothness")

    with col2:
        mean_compactness = st.number_input('Enter Mean Compactness', placeholder="Mean Compactness")

    with col1:
        mean_concavity = st.number_input('Enter Mean Concavity', placeholder="Mean Concavity")

    with col2:
        mean_concave_points = st.number_input('Enter Mean Concave Points', placeholder="Mean Concave Points")

    with col1:
        mean_symmetry = st.number_input('Enter Mean Symmetry', placeholder="Mean Symmetry")

    with col2:
        mean_fractal_dimension = st.number_input('Enter Mean Fractal Dimension', placeholder="Mean Fractal Dimension")

    with col1:
        radius_error = st.number_input('Enter Radius Error', placeholder="Mean Error")

    with col2:
        texture_error = st.number_input('Enter Texture Error', placeholder="Mean Texture Error")

    with col1:
        perimeter_error = st.number_input('Enter Perimeter Error', placeholder="Mean Perimeter Error")

    with col2:
        area_error = st.number_input('Enter Area Error', placeholder="Mean Area Error")

    with col1:
        smoothness_error = st.number_input('Enter Smoothness Error', placeholder="Mean Smoothness Error")

    with col2:
        compactness_error = st.number_input('Enter Compactness Error', placeholder="Mean Compactness Error")

    with col1:
        concavity_error = st.number_input('Enter Concavity Error', placeholder="Mean Concavity Error")

    with col2:
        concave_points_error = st.number_input('Enter Concave Points Error', placeholder="Mean Concave Points Error")

    with col1:
        symmetry_error = st.number_input('Enter Symmetry Error', placeholder="Mean Symmetry Error")

    with col2:
        fractal_dimension_error = st.number_input('Enter Fractal Dimension Error', placeholder="Mean Fractal Dimension Error")

    with col1:
        worst_radius = st.number_input('Enter Worst Radius', placeholder="Mean Worst Radius")

    with col2:
        worst_texture = st.number_input('Enter Worst Texture', placeholder="Mean Worst Texture")

    with col1:
        worst_perimeter = st.number_input('Enter Worst Perimeter', placeholder="Mean Worst Perimeter")

    with col2:
        worst_area = st.number_input('Enter Worst Area', placeholder="Mean Worst Area")

    with col1:
        worst_smoothness = st.number_input('Enter Worst Smoothness', placeholder="Mean Worst Smoothness")

    with col2:
        worst_compactness = st.number_input('Enter Worst Compactness', placeholder="Mean Worst Compactness")

    with col1:
        worst_concavity = st.number_input('Enter Worst Concavity', placeholder="Mean Worst Concavity")

    with col2:
        worst_concave_points = st.number_input('Enter Worst Concave Points', placeholder="Mean Worst Concave Points")

    with col1:
        worst_symmetry = st.number_input('Enter Worst Symmetry', placeholder="Mean Worst Symmetry")

    with col2:
        worst_fractal_dimension = st.number_input('Enter Worst Fractal Dimension', placeholder="Mean Worst Fractal Dimension")

    # Code for Prediction
    cancer_diagnosis = ''

    # Button for Prediction
    if st.button('Breast Cancer Test Result'):
        # Prepare user input for prediction
        user_input = [
            mean_radius, mean_texture, mean_perimeter, mean_area, mean_smoothness,
            mean_compactness, mean_concavity, mean_concave_points, mean_symmetry,
            mean_fractal_dimension, radius_error, texture_error, perimeter_error, area_error,
            smoothness_error, compactness_error, concavity_error, concave_points_error,
            symmetry_error, fractal_dimension_error, worst_radius, worst_texture, worst_perimeter,
            worst_area, worst_smoothness, worst_compactness, worst_concavity, worst_concave_points,
            worst_symmetry, worst_fractal_dimension
        ]

        # Convert input data to numpy array for prediction
        user_input = np.array(user_input).reshape(1, -1)

        # Predict breast cancer status using the pre-trained model
        cancer_prediction = breast_cancer_model.predict(user_input)

        # Output diagnosis based on prediction
        if cancer_prediction[0] == 1:
            cancer_diagnosis = 'The person is diagnosed with breast cancer'
        else:
            cancer_diagnosis = 'The person is not diagnosed with breast cancer'

    # Display the result
    st.success(cancer_diagnosis)

#Chronic Kidney Prediction Page
if(selected=='Chronic Kidney Disease Prediction'):
    st.title('Chronic Kidney Disease Prediction')

    # User Inputs
    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input('Age', min_value=1, max_value=120, placeholder="Age")
        blood_pressure = st.number_input('Blood Pressure (mm Hg)', placeholder="(mm Hg)")
        specific_gravity = st.selectbox('Specific Gravity', [1.005, 1.010, 1.015, 1.020, 1.025])
        albumin = st.number_input('Albumin Level (0-5)', min_value=0, max_value=5)
        sugar = st.number_input('Sugar Level (0-5)', min_value=0, max_value=5)
        red_blood_cells = st.selectbox('Red Blood Cells', ['normal', 'abnormal'])
        pus_cell = st.selectbox('Pus Cell', ['normal', 'abnormal'])
        pus_cell_clumps = st.selectbox('Pus Cell Clumps', ['present', 'notpresent'])
        bacteria = st.selectbox('Bacteria', ['present', 'notpresent'])
        blood_glucose_random = st.number_input('Random Blood Glucose (mg/dl)', placeholder="(mg/dl)")
        blood_urea = st.number_input('Blood Urea (mg/dl)', placeholder="(mg/dl)")
        serum_creatinine = st.number_input('Serum Creatinine (mg/dl)', placeholder="(mg/dl)")

    with col2:
        sodium = st.number_input('Sodium (mEq/L)', placeholder="(mEq/L)")
        potassium = st.number_input('Potassium (mEq/L)', placeholder="(mEq/L)")
        haemoglobin = st.number_input('Hemoglobin (g/dl)', placeholder="(g/dl)")
        packed_cell_volume = st.number_input('Packed Cell Volume (%)', placeholder="Cell Volume")
        white_blood_cell_count = st.number_input('WBC Count (cells/cumm)', placeholder="(cells/cumm)")
        red_blood_cell_count = st.number_input('RBC Count (millions/cumm)', placeholder="(millions/cumm)")
        hypertension = st.selectbox('Hypertension', ['yes', 'no'])
        diabetes_mellitus = st.selectbox('Diabetes Mellitus', ['yes', 'no'])
        coronary_artery_disease = st.selectbox('Coronary Artery Disease', ['yes', 'no'])
        appetite = st.selectbox('Appetite', ['good', 'poor'])
        peda_edema = st.selectbox('Pedal Edema', ['yes', 'no'])
        aanemia = st.selectbox('Anemia', ['yes', 'no'])

    # Convert categorical to numerical
    def convert_to_numeric(val, mapping):
        return mapping.get(val, -1)

    input_data = [
        age,
        blood_pressure,
        specific_gravity,
        albumin,
        sugar,
        convert_to_numeric(red_blood_cells, {'normal': 0, 'abnormal': 1}),
        convert_to_numeric(pus_cell, {'normal': 0, 'abnormal': 1}),
        convert_to_numeric(pus_cell_clumps, {'notpresent': 0, 'present': 1}),
        convert_to_numeric(bacteria, {'notpresent': 0, 'present': 1}),
        blood_glucose_random,
        blood_urea,
        serum_creatinine,
        sodium,
        potassium,
        haemoglobin,
        packed_cell_volume,
        white_blood_cell_count,
        red_blood_cell_count,
        convert_to_numeric(hypertension, {'no': 0, 'yes': 1}),
        convert_to_numeric(diabetes_mellitus, {'no': 0, 'yes': 1}),
        convert_to_numeric(coronary_artery_disease, {'no': 0, 'yes': 1}),
        convert_to_numeric(appetite, {'good': 0, 'poor': 1}),
        convert_to_numeric(peda_edema, {'no': 0, 'yes': 1}),
        convert_to_numeric(aanemia, {'no': 0, 'yes': 1}),
    ]

    # Prediction
    ckd_result = ''

    if st.button('Check CKD Risk'):
        # Convert to 2D array
        input_np = np.array(input_data).reshape(1, -1)
        ckd_prediction = chronic_kidney_model.predict(input_np)

        if ckd_prediction[0] == 1:
            ckd_result = 'The person is likely to have Chronic Kidney Disease.'
        else:
            ckd_result = 'The person is unlikely to have Chronic Kidney Disease.'

    st.success(ckd_result)

#Lung Cancer's Prediction Page
if(selected=='Lung Cancer Prediction'):
    st.title('Lung Cancer Prediction')

    # Collect input from user
    col1, col2 = st.columns(2)

    with col1:
        gender = st.selectbox("Select Gender", options=['Male (1)', 'Female (0)'])
        gender = 1 if gender == 'Male' else 0

    with col2:
        age = st.number_input('Enter Age', min_value=1, max_value=120, placeholder="Age")

    with col1:
        yellow_fingers = st.selectbox("Do You Have Yellow Fingers?", options=['No (1)', 'Yes (2)'])
        yellow_fingers = 2 if yellow_fingers == 'Yes' else 1

    with col2:
        anxiety = st.selectbox("Do You Have Anxiety?", options=['No (1)', 'Yes (2)'])
        anxiety = 2 if anxiety == 'Yes' else 1

    with col1:
        peer_pressure = st.selectbox("Affected by Peer Pressure?", options=['No (1)', 'Yes (2)'])
        peer_pressure = 2 if peer_pressure == 'Yes' else 1

    with col2:
        chronic_disease = st.selectbox("Do You Have Any Chronic Disease?", options=['No (1)', 'Yes (2)'])
        chronic_disease = 2 if chronic_disease == 'Yes' else 1

    with col1:
        fatigue = st.selectbox("Do You Experience Fatigue?", options=['No (1)', 'Yes (2)'])
        fatigue = 2 if fatigue == 'Yes' else 1

    with col2:
        allergy = st.selectbox("Do You Have Any Allergy?", options=['No (1)', 'Yes (2)'])
        allergy = 2 if allergy == 'Yes' else 1

    with col1:
        wheezing = st.selectbox("Do You Wheeze While Breathing?", options=['No (1)', 'Yes (2)'])
        wheezing = 2 if wheezing == 'Yes' else 1

    with col2:
        alcohol = st.selectbox("Do You Consume Alcohol?", options=['No (1)', 'Yes (2)'])
        alcohol = 2 if alcohol == 'Yes' else 1

    with col1:
        coughing = st.selectbox("Do You Have Persistent Coughing?", options=['No (1)', 'Yes (2)'])
        coughing = 2 if coughing == 'Yes' else 1

    with col2:
        short_breath = st.selectbox("Shortness of Breath?", options=['No (1)', 'Yes (2)'])
        short_breath = 2 if short_breath == 'Yes' else 1

    with col1:
        swallowing_difficulty = st.selectbox("Swallowing Difficulty?", options=['No (1)', 'Yes (2)'])
        swallowing_difficulty = 2 if swallowing_difficulty == 'Yes' else 1

    with col2:
        chest_pain = st.selectbox("Do You Have Chest Pain?", options=['No (1)', 'Yes (2)'])
        chest_pain = 2 if chest_pain == 'Yes' else 1

    # Prediction output
    lung_cancer_diagnosis = ''

    # Button
    if st.button('Lung Cancer Test Result'):
        user_data = np.array([
            gender, age, yellow_fingers, anxiety, 
            peer_pressure, chronic_disease, fatigue, 
            allergy, wheezing, alcohol, coughing, 
            short_breath, swallowing_difficulty , chest_pain
        ]).reshape(1, -1)

        lung_prediction = lung_cancer_model.predict(user_data)

        if lung_prediction[0] == 1:
            lung_cancer_diagnosis = "The person is likely to have Lung Cancer."
        else:
            lung_cancer_diagnosis = "The person is unlikely to have Lung Cancer."

    st.success(lung_cancer_diagnosis)

# Parkinson's Prediction Page
if (selected == "Parkinsons Disease Prediction"):
    
    # page title
    st.title("Parkinsons Disease Prediction")
    
    col1, col2, col3, col4, col5 = st.columns(5)  
    
    with col1:
        fo = st.text_input('MDVP:Fo(Hz)')
        
    with col2:
        fhi = st.text_input('MDVP:Fhi(Hz)')
        
    with col3:
        flo = st.text_input('MDVP:Flo(Hz)')
        
    with col4:
        Jitter_percent = st.text_input('MDVP:Jitter(%)')
        
    with col5:
        Jitter_Abs = st.text_input('MDVP:Jitter(Abs)')
        
    with col1:
        RAP = st.text_input('MDVP:RAP')
        
    with col2:
        PPQ = st.text_input('MDVP:PPQ')
        
    with col3:
        DDP = st.text_input('Jitter:DDP')
        
    with col4:
        Shimmer = st.text_input('MDVP:Shimmer')
        
    with col5:
        Shimmer_dB = st.text_input('MDVP:Shimmer(dB)')
        
    with col1:
        APQ3 = st.text_input('Shimmer:APQ3')
        
    with col2:
        APQ5 = st.text_input('Shimmer:APQ5')
        
    with col3:
        APQ = st.text_input('MDVP:APQ')
        
    with col4:
        DDA = st.text_input('Shimmer:DDA')
        
    with col5:
        NHR = st.text_input('NHR')
        
    with col1:
        HNR = st.text_input('HNR')
        
    with col2:
        RPDE = st.text_input('RPDE')
        
    with col3:
        DFA = st.text_input('DFA')
        
    with col4:
        spread1 = st.text_input('spread1')
        
    with col5:
        spread2 = st.text_input('spread2')
        
    with col1:
        D2 = st.text_input('D2')
        
    with col2:
        PPE = st.text_input('PPE')
        
    
    
    # code for Prediction
    parkinsons_diagnosis = ''
    
    # creating a button for Prediction    
    if st.button("Parkinson's Test Result"):
        parkinsons_prediction = parkinsons_model.predict([[fo, fhi, flo, Jitter_percent, Jitter_Abs, RAP, PPQ,DDP,Shimmer,Shimmer_dB,APQ3,APQ5,APQ,DDA,NHR,HNR,RPDE,DFA,spread1,spread2,D2,PPE]])                          
        
        if (parkinsons_prediction[0] == 1):
          parkinsons_diagnosis = "The person has Parkinson's disease"
        else:
          parkinsons_diagnosis = "The person does not have Parkinson's disease"
        
    st.success(parkinsons_diagnosis)