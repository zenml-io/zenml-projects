import streamlit as st
import pandas as pd 
from PIL import Image
import pickle 

from inference.predict import predict 

def main(): 
    st.title("End to End Customer Satisfaction Pipeline with ZenML")


    high_level_image = Image.open('_assets/high_level_overview.png')
    st.image(high_level_image, caption='High Level Pipeline')

    whole_pipeline_image = Image.open('_assets/training_and_deployment_pipeline_updated.png')
    
    st.markdown(""" 
    #### Problem Statement 
     The objective here is to predict the customer satisfaction score for a given order based on features like order status, price, payment, etc. I will be using [ZenML](https://zenml.io/) to build a production-ready pipeline to predict the customer satisfaction score for the next order or purchase.    """
    )
    st.image(whole_pipeline_image, caption='Whole Pipeline')
    st.markdown(""" 
    Above is a figure of the whole pipeline, we first ingest the data, clean it, train the model, and evaluate the model, and if data source changes or any hyperparameter values changes, deployment will be triggered, and (re) trains the model and if the model meets minimum accuracy requirement, the model will be deployed.
    """
    )

    st.markdown(""" 
    #### Description of Features 
    This app is designed to predict the customer satisfaction score for a given customer. You can input the features of the product listed below and get the customer satisfaction score. 
    | Models        | Description   | 
    | ------------- | -     | 
    | Payment Sequential | Customer may pay an order with more than one payment method. If he does so, a sequence will be created to accommodate all payments. | 
    | Payment Installments   | Number of installments chosen by the customer. |  
    | Payment Value |       Total amount paid by the customer. | 
    | Price |       Price of the product. |
    | Freight Value |    Freight value of the product.  | 
    | Product Name length |    Length of the product name. |
    | Product Description length |    Length of the product description. |
    | Product photos Quantity |    Number of product published photos |
    | Product weight measured in grams |    Weight of the product measured in grams. | 
    | Product length (CMs) |    Length of the product measured in centimeters. |
    | Product height (CMs) |    Height of the product measured in centimeters. |
    | Product width (CMs) |    Width of the product measured in centimeters. |
    """
    )
    payment_sequential = st.sidebar.slider("Payment Sequential") 
    payment_installments = st.sidebar.slider("Payment Installments")  
    payment_value = st.number_input("Payment Value") 
    price = st.number_input("Price")  
    freight_value = st.number_input("freight_value") 
    product_name_length = st.number_input("Product name length")  
    product_description_length = st.number_input("Product Description length") 
    product_photos_qty = st.number_input("Product photos Quantity ") 
    product_weight_g = st.number_input("Product weight measured in grams") 
    product_length_cm = st.number_input("Product length (CMs)") 
    product_height_cm = st.number_input("Product height (CMs)")  
    product_width_cm = st.number_input("Product width (CMs)")

    result = "" 
    if st.button("Predict"):
        with open('saved_model/model.pkl', 'rb') as handle:  
            model = pickle.load(handle) 
        result = predict(model, payment_sequential,
                        payment_installments,
                        payment_value,	
                        price,	
                        freight_value,	
                        product_name_length	,
                        product_description_length,	
                        product_photos_qty,	
                        product_weight_g,	
                        product_length_cm,
                        product_height_cm,
                        product_width_cm) 
        st.success('Your Customer Satisfactory rate(range between 0 - 5) with given product details is :-{}'.format(result))
    if st.button("Results"): 
        st.write("We have experimented with two ensemble and tree based models and compared the performance of each model. The results are as follows:")

        df = pd.DataFrame(
            { 
                'Models': ['LightGBM', 'Xgboost'], 
                'MSE': [1.804, 1.781], 
                'RMSE': [1.343, 1.335] 
            }
        )
        st.dataframe(df) 

        st.write("Following figure shows how important each feature is in the model that contributes to the target variable or contributes in predicting customer satisfaction rate.") 
        image = Image.open('_assets/feature_importance_gain.png') 
        st.image(image, caption='Feature Importance Gain')
    
if __name__=='__main__':
    main()


