from re import S
import streamlit as st
from inference.predict import predict 
from PIL import Image
import pandas as pd 

import pickle 
def main(): 
    st.title("End to End Customer Satisfaction Pipeline with ZenML")


    image = Image.open('_assets/high_level_overview.png')
    st.image(image, caption='High Level Pipeline')

    st.markdown("""
    #### Description of Features 
    This app is designed to predict the customer satisfaction score for a given customer.

    * Payment Sequential:-   Customer may pay an order with more than one payment method. If he does so, a sequence will be created to accommodate all payments.
    * Payment Installments:- Number of installments chosen by the customer.
    * Payment Value:-       Total amount paid by the customer.
    * Price:-              Price of the product.   
    * Frieght Value:-      Frieght value of the product.   
    * Product Name length:- Number of characters extracted from the product name.
    * Product Description length:- Number of characters extracted from the product description.
    * Product photos Quantity:- Number of product published photos
    * Product weight measured in grams:- Weight of the product measured in grams.
    * Product length (CMs):- product length measured in centimeters.
    * Product height measured in centimeters:- Height of the product measured in centimeters.
    * Product width (CMs):- Width of the product measured in centimeters.
    """
    )
    payment_sequential = st.sidebar.slider("Payment Sequential") 
    # payment_installments = st.sidebar.slider("Payment Installments") 
    payment_installments = st.sidebar.slider("Payment Installments")  
    # encoder the values 

    payment_value = st.number_input("Payment Value") 
    price = st.number_input("Price")  
    freight_value = st.number_input("freight_value") 
    product_name_lenght = st.number_input("Product name length")  
    product_description_lenght = st.number_input("Product Description length") 
    product_photos_qty = st.number_input("Product photos Quantity ") 
    product_weight_g = st.number_input("Product weight measured in grams") 
    product_length_cm = st.number_input("Product length (CMs)") 
    product_height_cm = st.number_input("Product height (CMs)")  
    product_width_cm = st.number_input("Product width (CMs)")

    result = "" 
    if st.button("Predict"):
        with open('model.pkl', 'rb') as handle:  
            model = pickle.load(handle) 
        result = predict(model, payment_sequential,
payment_installments,
payment_value,	
price,	
freight_value,	
product_name_lenght	,
product_description_lenght,	
product_photos_qty,	
product_weight_g,	
product_length_cm,
product_height_cm,
product_width_cm) 
        st.success('Your Customer Satisfactory rate(range between 0 - 5) with given product details is :-{}'.format(result))
    if st.button("Results"): 
        # make columns as Mod
        # Models	MSE	RMSE
        # LightGBM	1.804	1.343
        # XGboost	1.781	1.335
        # make df like above  
        st.write("We have experimented with 2 ensemble and tree based models and compared the performance of each model. The results are as follows:")

        df = pd.DataFrame(
            { 
                'Models': ['LightGBM', 'XGboost'], 
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


