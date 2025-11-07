import streamlit as st
import joblib
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
import lightgbm  # Required for the model to load
import traceback # Import traceback to print the full error

# --- 1. Load All Artifacts (Cached for speed) ---

@st.cache_resource
def load_artifacts():
    """Loads all necessary artifacts from the training notebook."""
    try:
        model = joblib.load('lgbm_calibrated_model.joblib')
        imputation_values = joblib.load('imputation_values.joblib')
        categorical_features = joblib.load('categorical_features.joblib')
        feature_columns = joblib.load('feature_columns.joblib')
        label_encoders = joblib.load('label_encoders.joblib')
        explainer = joblib.load('shap_explainer.joblib')
        return model, imputation_values, categorical_features, feature_columns, label_encoders, explainer
    
    # --- MODIFICATION: Catch ALL exceptions, not just FileNotFoundError ---
    except Exception as e:
        # Print the full error to the terminal
        print("---!! ERROR LOADING ARTIFACTS !!---")
        print(traceback.format_exc())
        print("-------------------------------------")
        
        # Display the error in the Streamlit app
        st.error(f"Error loading artifacts: {e}")
        st.code(traceback.format_exc())
        return (None,) * 6

# Load artifacts
model, IMPUTATION_VALUES, CAT_FEATURES, FEATURE_COLS, LABEL_ENCODERS, explainer = load_artifacts()

# --- MODIFICATION: Check if *all* artifacts loaded successfully ---
all_artifacts_loaded = all(v is not None for v in [model, IMPUTATION_VALUES, CAT_FEATURES, FEATURE_COLS, LABEL_ENCODERS, explainer])

# --- 2. App Title and Sidebar ---
st.set_page_config(layout="wide")
st.title("Loan Default Risk Dashboard 🏦")

# --- 3. Main App Logic ---
# Only run the rest of the app if loading was successful
if all_artifacts_loaded:
    st.write("Enter applicant details in the sidebar to get a real-time risk prediction and explanation.")
    st.sidebar.header("Applicant Information")

    # --- User Input Widgets ---
    int_rate = st.sidebar.slider("Interest Rate (%)", 5.0, 30.0, 12.5, 0.1)
    loan_amnt = st.sidebar.number_input("Loan Amount ($)", 500, 40000, 10000, 100)
    term = st.sidebar.select_slider("Loan Term", [" 36 months", " 60 months"])
    grade = st.sidebar.select_slider("Loan Grade", ["A", "B", "C", "D", "E", "F", "G"])
    annual_inc = st.sidebar.number_input("Annual Income ($)", 10000, 2000000, 75000, 100)
    dti = st.sidebar.slider("Debt-to-Income Ratio (DTI)", 0.0, 50.0, 15.0, 0.1)
    emp_length = st.sidebar.selectbox("Employment Length", ['10+ years', '2 years', '3 years', '< 1 year', '1 year', 
                                                           '4 years', '5 years', '6 years', '7 years', '8 years', '9 years', 'n/a'])
    home_ownership = st.sidebar.selectbox("Home Ownership", ['MORTGAGE', 'RENT', 'OWN', 'ANY'])
    purpose = st.sidebar.selectbox("Purpose", ['debt_consolidation', 'credit_card', 'home_improvement', 'other', 'major_purchase', 'small_business', 'car', 'medical', 'house', 'vacation'])
    credit_history_days = st.sidebar.number_input("Credit History (Days)", 300, 10000, 4500, 10)
    revol_util = st.sidebar.slider("Revolving Utilization (%)", 0.0, 100.0, 50.0, 0.1)
    inq_last_6mths = st.sidebar.number_input("Inquiries in Last 6 Months", 0, 10, 1, 1)

    # --- 4. Preprocessing and Prediction Function ---
    def predict_default_risk(input_data):
        """
        Takes user input, preprocesses it to match the model's training,
        and returns the default probability and the processed DataFrame for SHAP.
        """
        # 1. Create a DataFrame from inputs
        data_dict = IMPUTATION_VALUES.copy()
        
        # 2. Overwrite with user's inputs
        data_dict.update(input_data)
        
        # Create a single-row DataFrame in the correct column order
        df = pd.DataFrame([data_dict])[FEATURE_COLS]
        
        # 3. Create Interaction/Ratio Features (must match training)
        df['income_to_loan_ratio'] = df.get('annual_inc', 0) / (df.get('loan_amnt', 0) + 1)
        df['credit_utilization_ratio'] = df.get('revol_bal', 0) / (df.get('total_acc', 0) + 1)
        # --- FIX: Use .str.strip() and .str.replace() for Series ---
        df['term_in_months'] = df.get('term', ' 36 months').str.strip().str.replace(' months', '').astype(float)
        df['total_payment'] = df.get('term_in_months', 36.0) * df.get('installment', 0)
        df['payment_to_income_ratio'] = df.get('installment', 0) / (df.get('annual_inc', 0) + 1)
        df['open_acc_to_total_acc'] = df.get('open_acc', 0) / (df.get('total_acc', 0) + 1)
        df['util_x_dti'] = df.get('revol_util', 0) * df.get('dti', 0)
        
        # 4. Encode Categorical Features
        df_processed = df.copy()
        for col in CAT_FEATURES:
            if col in df_processed.columns:
                le = LABEL_ENCODERS[col]
                val = str(df_processed[col].iloc[0])
                if val in le.classes_:
                    df_processed[col] = le.transform([val])[0]
                else:
                    df_processed[col] = -1
                df_processed[col] = df_processed[col].astype('category')
            
        # 5. Predict
        probability = model.predict_proba(df_processed)[:, 1][0]
        return probability, df_processed

    # --- 5. Display Prediction and Explanation ---
    if st.sidebar.button("Calculate Default Risk"):
        # 1. Collect inputs
        user_input = {
            'int_rate': int_rate,
            'loan_amnt': loan_amnt,
            'term': term,
            'grade': grade,
            'annual_inc': annual_inc,
            'dti': dti,
            'emp_length': emp_length,
            'home_ownership': home_ownership,
            'purpose': purpose,
            'credit_history_days': credit_history_days,
            'revol_util': revol_util,
            'inq_last_6mths': inq_last_6mths
        }
        
        # 2. Get prediction
        prob, processed_df = predict_default_risk(user_input)
        
        # 3. Display Risk
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.header("Risk Profile")
            st.metric("Probability of Default", f"{prob:.2%}")
            
            # Use the optimal threshold we found (or a business-defined one)
            optimal_threshold = 0.3 # <-- REPLACE with your 'best_threshold'
            
            if prob > optimal_threshold:
                st.error("High Risk (Above Threshold)")
                st.write(f"This applicant's risk ({prob:.2%}) is above the {optimal_threshold:.0%} F1-Score threshold. **Rejection Recommended.**")
            else:
                st.success("Low Risk (Below Threshold)")
                st.write(f"This applicant's risk ({prob:.2%}) is below the {optimal_threshold:.0%} threshold. **Approval Recommended.**")
            
            st.subheader("Key Applicant Inputs")
            st.json(user_input)

        # 4. Display SHAP Explanation
        with col2:
            st.header("Top Risk Factors (SHAP Explanation)")
            
            shap_values_list = explainer.shap_values(processed_df)
            
            if isinstance(explainer.expected_value, (list, np.ndarray)):
                expected_value = explainer.expected_value[1]
            else:
                expected_value = explainer.expected_value
                
            if isinstance(shap_values_list, list) and len(shap_values_list) == 2:
                shap_values_class_1 = shap_values_list[1]
            else:
                shap_values_class_1 = shap_values_list

            shap_explanation = shap.Explanation(
                values=shap_values_class_1[0, :],
                base_values=expected_value,
                data=processed_df.iloc[0, :],
                feature_names=processed_df.columns.tolist()
            )
            
            fig, ax = plt.subplots(figsize=(10, 10))
            shap.waterfall_plot(shap_explanation, max_display=15, show=False)
            plt.tight_layout()
            st.pyplot(fig)

    else:
        st.info("Please fill in the applicant details on the left and click 'Calculate Default Risk'.")

# This 'else' will now catch if the artifacts failed to load
else:
    st.error("Application could not load model artifacts. Please check the file paths and ensure all .joblib files are in the same directory as app.py.")