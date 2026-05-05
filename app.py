import streamlit as st
import pickle
import pandas as pd

# Load
lr = pickle.load(open('lr.pkl', 'rb'))
rf = pickle.load(open('rf.pkl', 'rb'))
xgb = pickle.load(open('xgb.pkl', 'rb'))
columns = pickle.load(open('columns.pkl', 'rb'))
results_df = pickle.load(open('results.pkl', 'rb'))

st.title("💰 Tip Prediction App (Full Model)")

st.subheader("📊 Model Comparison")
st.dataframe(results_df)

model_choice = st.selectbox("Choose Model", results_df["Model"])

bill = st.number_input("Total Bill", min_value=0.0)
size = st.number_input("Number of People", min_value=1)

sex = st.selectbox("Sex", ["Male", "Female"])
smoker = st.selectbox("Smoker", ["Yes", "No"])

if st.button("Predict Tip"):
    input_dict = {
        'total_bill': bill,
        'size': size,
        'sex_Male': 1 if sex == "Male" else 0,
        'smoker_Yes': 1 if smoker == "Yes" else 0,
    }

    input_df = pd.DataFrame([input_dict])
    input_df = input_df.reindex(columns=columns, fill_value=0)

    if model_choice == "Linear Regression":
        model = lr
    elif model_choice == "Random Forest":
        model = rf
    else:
        model = xgb

    prediction = model.predict(input_df)
    st.success(f"Predicted Tip: $ {prediction[0]:.2f}")
