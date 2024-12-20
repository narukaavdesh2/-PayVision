import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
import streamlit as st

# Load dataset
salary_dataset = pd.read_csv('./Salary Data.csv')

# Data preprocessing
salary_dataset.dropna(inplace=True)
salary_dataset['Salary'] = pd.to_numeric(salary_dataset['Salary'], errors='coerce')
salary_dataset['Years of Experience'] = pd.to_numeric(salary_dataset['Years of Experience'], errors='coerce')
salary_dataset['Age'] = pd.to_numeric(salary_dataset['Age'], errors='coerce')
salary_dataset.drop_duplicates(inplace=True)
salary_dataset.reset_index(inplace=True, drop=True)

# Encoding categorical variables
salary_dataset.replace({'Education Level': {"Bachelor's": 0, "Master's": 1, "PhD": 2}}, inplace=True)

# Main Application
st.title("PayVision: Salary Insights and Prediction App")

# Create a single-page layout with sections using divs and spans
with st.container():
    st.markdown("""<div style='padding: 20px; border: 1px solid #ddd; border-radius: 8px;'>""", unsafe_allow_html=True)
    st.header("Data Overview and Insights")
    
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Dataset Snapshot")
        st.write(salary_dataset.head())

    with col2:
        st.subheader("Summary Statistics")
        st.write(salary_dataset.describe())
        st.write(f"*Number of Duplicates:* {salary_dataset.duplicated().sum()}")

    st.markdown("</div>", unsafe_allow_html=True)

with st.container():
    st.markdown("""<div style='padding: 20px; margin-top: 20px; border: 1px solid #ddd; border-radius: 8px;'>""", unsafe_allow_html=True)
    st.header("Visualizations")
    
    # Visualizations for Salary Insights
    st.subheader("Salary Insights by Gender and Education Level")
    
    col1, col2 = st.columns(2)

    with col1:
        salary_by_gender = salary_dataset.groupby("Gender")["Salary"].mean()
        fig = px.bar(x=salary_by_gender.index, y=salary_by_gender.values, 
                     labels={'x': 'Gender', 'y': 'Mean Salary'}, title='Mean Salary by Gender')
        st.plotly_chart(fig)

    with col2:
        salary_by_level = salary_dataset.groupby("Education Level")["Salary"].mean()
        fig = px.bar(x=salary_by_level.index.map({0: "Bachelor's", 1: "Master's", 2: "PhD"}), 
                     y=salary_by_level.values, labels={'x': 'Education Level', 'y': 'Mean Salary'},
                     title='Mean Salary by Education Level')
        st.plotly_chart(fig)

    # Visualizations for Years of Experience and Age Groups
    st.subheader("Salary Insights by Years of Experience and Age Groups")
    
    col1, col2 = st.columns(2)

    with col1:
        def groupping_exp(exp):
            if exp <= 5:
                return "0-5 years"
            elif exp <= 10:
                return "6-10 years"
            elif exp <= 15:
                return "11-15 years"
            elif exp <= 20:
                return "16-20 years"
            else:
                return "20+"

        salary_by_exp = salary_dataset.groupby(salary_dataset["Years of Experience"].apply(groupping_exp))["Salary"].mean()
        fig = px.bar(x=salary_by_exp.index, y=salary_by_exp.values, 
                     labels={'x': 'Years of Experience', 'y': 'Mean Salary'},
                     title='Mean Salary by Years of Experience')
        st.plotly_chart(fig)

    with col2:
        def groupping_age(age):
            if age >= 20 and age <= 25:
                return "20-25 years"
            elif age <= 30:
                return "25-30 years"
            elif age <= 35:
                return "30-35 years"
            elif age <= 40:
                return "35-40 years"
            elif age <= 45:
                return "40-45 years"
            elif age <= 50:
                return "45-50 years"
            else:
                return "50+"

        salary_by_age = salary_dataset.groupby(salary_dataset["Age"].apply(groupping_age))["Salary"].mean()
        fig = px.bar(x=salary_by_age.index, y=salary_by_age.values, 
                     labels={'x': 'Age Groups', 'y': 'Mean Salary'},
                     title='Mean Salary by Age Groups')
        st.plotly_chart(fig)

    st.markdown("</div>", unsafe_allow_html=True)

with st.container():
    st.markdown("""<div style='padding: 20px; margin-top: 20px; border: 1px solid #ddd; border-radius: 8px;'>""", unsafe_allow_html=True)
    st.header("Model Training and Prediction")

    # Feature selection
    X = salary_dataset.drop(['Gender', 'Job Title', 'Salary'], axis=1)
    Y = salary_dataset['Salary']

    # Split data
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=10)

    # Train model
    with st.spinner('Training Linear Regression Model...'):
        lin_model = LinearRegression()
        lin_model.fit(X_train, Y_train)
    st.success('Model Training Complete!')

    # Evaluate model
    training_data_predict = lin_model.predict(X_train)
    error_score = metrics.r2_score(Y_train, training_data_predict)
    st.metric("Training R-Squared Error", f"{error_score:.2f}")

    test_data_predict = lin_model.predict(X_test)
    test_error_score = metrics.r2_score(Y_test, test_data_predict)
    st.metric("Test R-Squared Error", f"{test_error_score:.2f}")

    # Visualize results
    col1, col2 = st.columns(2)
    with col1:
        st.write("### Actual vs Predicted (Training Data)")
        fig, ax = plt.subplots()
        plt.scatter(Y_train, training_data_predict)
        plt.xlabel("Actual")
        plt.ylabel("Predicted")
        plt.title("Actual vs Predicted Salary (Training)")
        st.pyplot(fig)

    with col2:
        st.write("### Actual vs Predicted (Test Data)")
        fig, ax = plt.subplots()
        plt.scatter(Y_test, test_data_predict)
        plt.xlabel("Actual")
        plt.ylabel("Predicted")
        plt.title("Actual vs Predicted Salary (Test)")
        st.pyplot(fig)

    st.subheader("Predict Salary")

    # Input for prediction
    years_experience = st.number_input("Years of Experience", min_value=0, value=0)
    education_level = st.selectbox("Education Level", ["Bachelor's", "Master's", "PhD"])
    age = st.number_input("Age", min_value=18, value=25)

    # Convert inputs and predict
    education_map = {"Bachelor's": 0, "Master's": 1, "PhD": 2}
    user_data = pd.DataFrame({
        'Years of Experience': [years_experience],
        'Education Level': [education_map[education_level]],
        'Age': [age]
    })

    # Ensure that user_data columns are in the same order as X_train columns
    user_data = user_data[X_train.columns]

    if st.button("Predict Salary"):
        predicted_salary = lin_model.predict(user_data)
        st.write(f"Predicted Salary: ${predicted_salary[0]:,.2f}")

    st.markdown("</div>", unsafe_allow_html=True)
