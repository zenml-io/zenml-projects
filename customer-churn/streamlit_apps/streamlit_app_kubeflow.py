import streamlit as st
from zenml.repository import Repository


def main():
    st.title("Predicting whether the customer will churn or not before they even did it")

    st.markdown(
        """ 
    #### Problem Statement 
    For a given customer's historical data, we are asked to predict whether a customer will stop using a company's product or not. We will be using the [Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn?datasetId=13996&sortBy=voteCount) dataset for building an end to end production-grade machine learning system that can predict whether the customer will stay loyal or not. 
    To achieve this in a real-world scenario, we will be using [ZenML](https://zenml.io/) to build a production-ready pipeline that predicts whether a customer will churn or not ahead of time.
    The purpose of this repository is to demonstrate how [ZenML](https://github.com/zenml-io/zenml) empowers your business to build and deploy machine learning pipelines in a multitude of ways:

    - By offering you a framework or template to develop within.
    - By integrating with popular and useful tools like Kubeflow, Seldon Core, `facets`, and more.
    - By allowing you to build and deploy your machine learning pipelines easily using a modern MLOps framework.

    """
    )

    st.markdown(
        """ 
    #### Description of Features 
    This app is designed to predict whether customer will churn the company or not. You can input the features of the product listed below and get the prediction. 
    - Customers who left within the last month:- the column is called Churn
    - Services that each customer has signed up for:-  phone, multiple lines, internet, online security, online backup, device protection, tech support, and streaming TV and movies
    - Customer account information:- how long they have been a customer, contract, payment method, paperless billing, monthly charges, and total charges
    - Demographic info about customers:- gender, age range, and if they have partners and dependents

    """
    )
    customer_id = st.number_input("Customer ID")
    gender = st.number_input("Gender")
    SeniorCitizen = st.number_input("Senior Citizen")
    Partner = st.number_input("Partner")
    Dependents = st.number_input("Dependents")
    tenure = st.number_input("tenure")
    PhoneService = st.number_input("PhoneService")
    MultipleLines = st.number_input("MultipleLines")
    InternetService = st.number_input("InternetService")
    OnlineSecurity = st.number_input("OnlineSecurity")
    OnlineBackup = st.number_input("OnlineBackup")
    DeviceProtection = st.number_input("DeviceProtection")
    TechSupport = st.number_input("TechSupport")
    StreamingTV = st.number_input("StreamingTV")
    StreamingMovies = st.number_input("StreamingMovies")
    Contract = st.number_input("Contract")
    PaperlessBilling = st.number_input("PaperlessBilling")
    PaymentMethod = st.number_input("PaymentMethod")
    MonthlyCharges = st.number_input("MonthlyCharges")
    TotalCharges = st.number_input("TotalCharges")

    if st.button("Predict"):
        repo = Repository()
        p = repo.get_pipeline("training_pipeline")
        last_run = p.runs[-1]
        trainer_step = last_run.get_step("model_trainer")
        model = trainer_step.output.read()

        pred = [
            [
                customer_id,
                gender,
                SeniorCitizen,
                Partner,
                Dependents,
                tenure,
                PhoneService,
                MultipleLines,
                InternetService,
                OnlineSecurity,
                OnlineBackup,
                DeviceProtection,
                TechSupport,
                StreamingTV,
                StreamingMovies,
                Contract,
                PaperlessBilling,
                PaymentMethod,
                MonthlyCharges,
                TotalCharges,
            ]
        ]
        pred = model.predict(pred)
        st.write(pred)


if __name__ == "__main__":
    main()
