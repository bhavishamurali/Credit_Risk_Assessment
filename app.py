import streamlit as st
import pickle
import pandas as pd

# Set up a wide, attractive page layout
st.set_page_config(
    page_title="Credit Risk AI Dashboard", 
    page_icon="💳", 
    layout="wide"
)

# Header Section with an appealing design
st.title("💳 Intelligent Credit Risk Assessment Portal")
st.markdown("### *Automated Financial Analytics & Loan Default Risk Predictor*")
st.write("Adjust the borrower criteria parameters in the side panel to dynamically compute risk percentages and check credit underwriting eligibility.")
st.markdown("---")

@st.cache_resource
def load_risk_model():
    with open('outputs/credit_model.pkl', 'rb') as f:
        return pickle.load(f)

try:
    model = load_risk_model()
    
    # Split the screen layout into a clean Sidebar (Inputs) and Main Panel (Results)
    with st.sidebar:
        st.header("📋 Applicant Profile")
        st.write("Modify the financial variables below:")
        
        # Financial Input Sliders clustered cleanly in the sidebar
        credit_score = st.slider("FICO Credit Score", 300, 850, 650, help="Standardized credit bureau score distribution.")
        annual_income = st.slider("Annual Income ($k)", 20, 150, 60, help="Total gross applicant earnings per year in thousands.")
        dti_ratio = st.slider("Debt-to-Income (DTI) Ratio", 0.0, 1.0, 0.3, step=0.05, help="Monthly debt payments divided by monthly gross income.")
        employment_years = st.slider("Employment History (Years)", 0, 20, 4, help="Years spent at current workplace.")
        
        st.markdown("---")
        evaluate_btn = st.button("🚀 Execute Risk Analysis", use_container_width=True)

    # Bundle inputs for prediction execution
    input_data = pd.DataFrame([{
        'Credit_Score': credit_score,
        'Annual_Income_K': annual_income,
        'Debt_to_Income_Ratio': dti_ratio,
        'Employment_Years': employment_years
    }])

    # Main dashboard section displays visual layout cards
    st.subheader("📊 Live Underwriting Analysis Dashboard")
    
    # Visual metric display cards to make the data pop out immediately
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Selected FICO Score", f"⭐ {credit_score}")
    col2.metric("Stated Income", f"${annual_income},000")
    col3.metric("DTI Leverage", f"📊 {int(dti_ratio * 100)}%")
    col4.metric("Job Stability", f"⏱️ {employment_years} Yrs")

    st.markdown("#### **System Output Evaluation Matrix**")

    # Run processing loop on button press or profile adjustment
    if evaluate_btn or (credit_score != 650):  # Triggers automatically or via button
        prediction = model.predict(input_data)[0]
        prob = model.predict_proba(input_data)[0][1] * 100
        
        # Display large attractive custom containers based on status
        if prediction == 1:
            with st.container(border=True):
                st.error("### 🚨 Underwriting Risk Verdict: REJECT / HIGH RISK")
                st.progress(int(prob))
                st.write(f"The structural pipeline model reports an alarming **{prob:.1f}% probability of default** based on the specified parameters.")
        else:
            with st.container(border=True):
                st.success("### 🎯 Underwriting Risk Verdict: APPROVED / CLEAN CREDIT")
                st.progress(int(prob))
                st.write(f"The candidate is safe to clear. The structural pipeline reports a comfortable, low risk factor of only **{prob:.1f}% probability of default**.")
    else:
        st.info("💡 Adjust the parameters in the sidebar pane or click 'Execute Risk Analysis' to generate custom validation results.")
            
except Exception as e:
    st.error("Please run 'python train_model.py' inside your terminal execution space first to assemble structural pkl weights maps.")