import streamlit as st
from src.prediction import Insurance_Prediction

st.title("Insurance Prediction")
st.write("This project predicts medical insurance costs using machine learning based on factors such as age, BMI, number of children, smoking status, and region. The model is trained using regression techniques and integrated with a simple Streamlit web interface where users can input their details to estimate their insurance charges.")

Age=st.number_input("Enter age:",min_value=1,max_value=100)
Annual_Income_LPA=st.number_input("Annual_Income_LPA:",min_value=1,max_value=100)
Policy_Term_Years=st.number_input("Policy_Term_Years:",min_value=1,max_value=100)
Sum_Assured_Lakhs=st.number_input("Sum_Assured_Lakhs:",min_value=1,max_value=100)

if st.button("Predict"):
    model=Insurance_Prediction()
    result=model.prediction(Age,Annual_Income_LPA,Policy_Term_Years,Sum_Assured_Lakhs)
    st.success(f"Predicted Insurance Cost: ₹ {result[0]:.2f Lakhs}")