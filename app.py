import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
import dice_ml
#from dice_ml import Data, Model, Dice
from sklearn.pipeline import Pipeline
import streamlit.components.v1 as components
#from IPython.display import display

# --- 1. SETTINGS & ASSETS ---
st.set_page_config(page_title="Interpretaple Diabetes Mellitus Prediction and Personalized Counterfactual Visualization Model ", layout="wide")
st.markdown("""
    <p style='font-size: 18px; color: gray; margin-top: -20px;'>
    This model was trained based on the Iraqi Diabetes dataset. https://data.mendeley.com/datasets/wj9rwkp9c2/1 
    </p>
    """, unsafe_allow_html=True)
@st.cache_resource
def load_assets():
    model = joblib.load('random_model.pkl')
    scaler = joblib.load('scaler.pkl')
    # Load training data for DiCE and Gini Importance
    pipeline = joblib.load('diabetes_model_pipeline.pkl')
    train_df = pd.read_csv('diabetes_cleaned_final2.csv')
    return model, scaler, pipeline, train_df

rf_model, scaler, pipeline, train_df = load_assets()
feature_names = ['AGE', 'Urea', 'Cr', 'HbA1c', 'Chol', 'TG', 'HDL', 'LDL', 'VLDL', 'BMI', 'Gender_Encoded']


model_pipeline = Pipeline([
    ('scaler', scaler),
    ('rf', rf_model)
])
# Initialize Session State to track if prediction has been made
if 'prediction_done' not in st.session_state:
    st.session_state.prediction_done = False

# --- 2. FRONT PAGE: 11 FACTORS INPUT ---
st.title("🩺 Interpretaple Diabetes Mellitus Prediction and Personalized Counterfactual Visualization Model")
st.markdown("### Patient Biomarker Entry")
st.write("Enter the raw clinical values below to begin the analysis.")

with st.container():
    col1, col2, col3 = st.columns(3)
    
    with col1:
        age = st.number_input("Age", value=45)#45
        urea = st.number_input("Urea  mmol/L", value=5.0)#5.0
        cr = st.number_input("Cr (Creatinine) mg/dL", value=80.0)#80.0
        hba1c = st.number_input("HbA1c (%)", value=6.0)#6.0
    
    with col2:
        chol = st.number_input("Cholesterol  mmol/L", value=4.5)#4.5
        tg = st.number_input("TG (Triglycerides) mmol/L", value=1.5)#1.5
        hdl = st.number_input("HDL mmol/L", value=1.2)#1.2
        ldl = st.number_input("LDL mmol/L", value=3.0)#3.0

    with col3:
        vldl = st.number_input("VLDL mmol/L", value=0.6)#0.6
        bmi = st.number_input("BMI", value=26.0)#26.0
        gender = st.selectbox("Gender", options=["Male", "Female"])#1
        gender_encoded = 1 if gender == "Male" else 0

# Prepare Data
raw_input = pd.DataFrame(
    [[age, urea, cr, hba1c, chol, tg, hdl, ldl, vldl, bmi, gender_encoded]],
    columns=feature_names
)


#scaled_input = pd.DataFrame(scaler.transform(raw_input), columns=feature_names)

# --- 3. THE PREDICTION TRIGGER ---
if st.button("🚀 Run Diabetes Prediction"):
    st.session_state.prediction_done = True

# --- 4. ANALYSIS SECTION (Only visible after clicking button) ---
if st.session_state.prediction_done:
    st.divider()
    
    # Run Model
   # pred = rf_model.predict(scaled_input)[0]
    #probs = rf_model.predict_proba(scaled_input)[0]
    pred = model_pipeline.predict(raw_input)[0]
    probs = model_pipeline.predict_proba(raw_input)[0]
    labels = ["Healthy", "Pre-Diabetic", "Diabetic"]
    
    # TABS FOR NAVIGATION
    tab_global, tab_shap, tab_dice, tab_dice2, tab_dice3, tab_eval = st.tabs([
        "🌍 Global Analysis", "🔍 SHAP Local", "🛠️ Counterfactual for Healthy(0)","🛠️ Counterfactual for Prediabetes(1)","🛠️ Counterfactual for Diabetes(2)", "📋 Trust Measuring Questionnaires"
    ])

    # --- TAB: GLOBAL ANALYSIS ---
    with tab_global:
        st.subheader("Global Prediction Results")
        
        # Result Metric
        color = "green" if pred == 0 else "orange" if pred == 1 else "red"
        st.markdown(f"### Predicted Status: <span style='color:{color}'>{labels[int(pred)]}</span>", unsafe_allow_html=True)
        
        c1, c2, c3 = st.columns(3)
        c1.metric("Healthy", f"{probs[0]*100:.1f}%")
        c2.metric("Pre-Diabetic", f"{probs[1]*100:.1f}%")
        c3.metric("Diabetic", f"{probs[2]*100:.1f}%")

        st.divider()
        st.subheader("Gini Feature Importance (Model-Wide)")
        
        # Calculate Gini Importance
        importances = rf_model.feature_importances_
        indices = np.argsort(importances)
        
        fig_gini, ax_gini = plt.subplots()
        ax_gini.barh(range(len(indices)), importances[indices], color='skyblue', align='center')
        ax_gini.set_yticks(range(len(indices)))
        ax_gini.set_yticklabels([feature_names[i] for i in indices])
        ax_gini.set_xlabel('Relative Importance')
        st.pyplot(fig_gini)
        

    # --- TAB: SHAP LOCAL ANALYSIS ---
    with tab_shap:
        st.subheader("SHAP Local Explanation (Waterfall)")
        
        # Create Explainer on the fly (most stable for deployment)
        explainer = shap.TreeExplainer(rf_model)
        shap_values = explainer.shap_values(raw_input)
        
        # Extract 1D array for predicted class
        class_idx = int(pred)
        # Handle list (multi-class) or array
        current_shap = shap_values[class_idx][0] if isinstance(shap_values, list) else shap_values[0, :, class_idx]
        
        exp = shap.Explanation(
            values=current_shap,
            base_values=explainer.expected_value[class_idx],
            data=raw_input.iloc[0].values,
            feature_names=feature_names
        )
        
        fig_shap, ax_shap = plt.subplots()
        shap.plots.waterfall(exp, show=False)
        st.pyplot(fig_shap)
        st.info("Red bars show factors increasing the risk of this diagnosis; Blue bars show factors decreasing it.")
        

          # --- TAB 3: TRANSITION & RISK FORECASTING ---
    with tab_dice:
        st.subheader("DiCE Counterfactual Intervention")
        st.write("What changes are needed to reach a 'Healthy' status?")
        
        # DiCE Setup
        # We use a copy of training data for DiCE context
        d_data = dice_ml.Data(dataframe=train_df, continuous_features=['AGE', 'Urea', 'Cr', 'HbA1c', 'Chol', 'TG', 'HDL', 'LDL', 'VLDL', 'BMI', 'Gender_Encoded'], outcome_name='Class_Encoded')
        d_model = dice_ml.Model(model=model_pipeline, backend="sklearn")
        exp_dice = dice_ml.Dice(d_data, d_model, method="random")
        
        current_pred = int(model_pipeline.predict(raw_input)[0])
        original_age = float(raw_input['AGE'].iloc[0])
        # Define Age constraint: Cannot go below current age
        # Merge all constraints into one dictionary
        #clinical_constraints = {
        # 'HbA1c': [4.0, 15.0], 
        #'BMI': [18.0, 50.0],
        #'Age': [original_age, 100.0]  # Ensures the CF doesn't suggest getting younger
        #}
        age_constraint = [original_age, 100.0] 
        hba1c_constraint = [4.0, 15.0] 
        bmi_constraint = [18.0, 30.0] 
        
        st.write(current_pred)
        if current_pred != 0:
            # Generate 2 counterfactuals to show how to reach Class 0 (Healthy)
            cf = exp_dice.generate_counterfactuals(raw_input, total_CFs=2, desired_class=0, features_to_vary=['HbA1c', 'BMI', 'TG', 'Chol'], permitted_range={'AGE': age_constraint,'HbA1c': hba1c_constraint,'BMI': bmi_constraint})
            # 1. DISPLAY ORIGINAL DATA
            st.markdown("### **Step 1: Current Patient Profile**")
            st.write("This is the patient's actual recorded data:")
            st.dataframe(raw_input)
            st.divider() # Visual separator
            st.write("To become **Healthy**, the patient should aim for these biomarker changes:")
            st.dataframe(cf.cf_examples_list[0].final_cfs_df)
        else:
           st.success("Patient is already in the same Class. No intervention needed.")
      # --- TAB 3: TRANSITION & RISK FORECASTING ---
    
    
    with tab_dice2:
        st.subheader("DiCE Counterfactual Intervention")
        st.write("What changes are needed to reach a 'Prediabetic' status?")
        
        # DiCE Setup
        # We use a copy of training data for DiCE context
        d_data = dice_ml.Data(dataframe=train_df, continuous_features=['AGE', 'Urea', 'Cr', 'HbA1c', 'Chol', 'TG', 'HDL', 'LDL', 'VLDL', 'BMI', 'Gender_Encoded'], outcome_name='Class_Encoded')
        d_model = dice_ml.Model(model=model_pipeline, backend="sklearn")
        exp_dice = dice_ml.Dice(d_data, d_model, method="genetic")
        current_pred = int(model_pipeline.predict(raw_input)[0])
        original_age = float(raw_input['AGE'].iloc[0])
        # Define Age constraint: Cannot go below current age
        age_constraint = [original_age, 100.0] 
        hba1c_constraint = [4.0, 15.0] 
        bmi_constraint = [18.0, 30.0] 
        st.write(current_pred)
        if current_pred != 1:
            # Generate 2 counterfactuals to show how to reach Class 0 (Healthy)
            cf = exp_dice.generate_counterfactuals(raw_input, total_CFs=2, desired_class=1, features_to_vary=['HbA1c', 'BMI', 'TG', 'Chol'], permitted_range={'AGE': age_constraint,'HbA1c': hba1c_constraint,'BMI': bmi_constraint})
            # 1. DISPLAY ORIGINAL DATA
            st.markdown("### **Step 1: Current Patient Profile**")
            st.write("This is the patient's actual recorded data:")
            st.dataframe(raw_input)
            st.divider() # Visual separator
            st.write("To become **Prediabetic**, the patient should aim for these biomarker changes:")
            st.dataframe(cf.cf_examples_list[0].final_cfs_df)
        else:
            st.success("Patient is already in the same Class. No intervention needed.")
    
    # --- TAB: INTERVIEW / GOOGLE FORM ---
    with tab_dice3:
        st.subheader("DiCE Counterfactual Intervention")
        st.write("What changes are needed to analysis a risk of 'Diabetes' status?")
        
        # DiCE Setup
        # We use a copy of training data for DiCE context
        d_data = dice_ml.Data(dataframe=train_df, continuous_features=['AGE', 'Urea', 'Cr', 'HbA1c', 'Chol', 'TG', 'HDL', 'LDL', 'VLDL', 'BMI', 'Gender_Encoded'], outcome_name='Class_Encoded')
        d_model = dice_ml.Model(model=model_pipeline, backend="sklearn")
        exp_dice = dice_ml.Dice(d_data, d_model, method="genetic")
        current_pred = int(model_pipeline.predict(raw_input)[0])
        original_age = float(raw_input['AGE'].iloc[0])
        # Define Age constraint: Cannot go below current age
        age_constraint = [original_age, 100.0]
        hba1c_constraint = [4.0, 15.0] 
        bmi_constraint = [18.0, 30.0]
        st.write(current_pred)
        if current_pred != 2:
            # Generate 2 counterfactuals to show how to reach Class 0 (Healthy)
            cf = exp_dice.generate_counterfactuals(raw_input, total_CFs=2, desired_class=2, features_to_vary=['HbA1c', 'BMI', 'TG', 'Chol'], permitted_range={'AGE': age_constraint,'HbA1c': hba1c_constraint,'BMI': bmi_constraint})
           # 1. DISPLAY ORIGINAL DATA
            st.markdown("### **Step 1: Current Patient Profile**")
            st.write("This is the patient's actual recorded data:")
            st.dataframe(raw_input)
            st.divider() # Visual separator
            
            # 1. Get the original data and the generated counterfactuals
            #query_instance = cf.cf_examples_list[0].test_instance_df
            #cf_df = cf.cf_examples_list[0].final_cfs_df

            # 2. Identify only the columns where the counterfactual value is different from the original
            # We use .iloc[0] because query_instance is a single-row dataframe
            #changed_cols = [col for col in cf_df.columns if not (cf_df[col] == query_instance[col].iloc[0]).all()]

            # 3. Display the selective dataframe
            #st.subheader("Selective Counterfactuals: Minimal Changes Required")
            #st.dataframe(cf_df[changed_cols])
            st.write("Inorder not to become **Diabetes**, the patient should be careful for these biomarker changes:")
            st.dataframe(cf.cf_examples_list[0].final_cfs_df)
            #st.dataframe(cf.visualize_as_dataframe(show_only_changes=True))
        else:
            st.success("Patient is already in the same Class. No intervention needed.")
    
    # --- TAB: INTERVIEW / GOOGLE FORM ---
    with tab_eval:
        st.subheader("Trust Evaluation Form")
        # Replace with your actual Google Form Embed Link
        google_form_url = "https://docs.google.com/forms/d/e/1FAIpQLSfhMAcwF3SrOZLQh_kBnJ43HyIrBjk-8nt870UbCgkQv9b8Jw/viewform?usp=dialog"
        components.iframe(google_form_url, height=800, scrolling=True)


    # --- DEBUGGING LINES ---
#st.write("Columns found in CSV:", train_df.columns.tolist())
#st.write("Columns expected as Continuous:", feature_names[:-1])
#st.write("Target column expected:", 'Analysis')
# -----------------------


#d_data = Data(dataframe=train_df, continuous_features=feature_names[:-1], outcome_name='Analysis')
















