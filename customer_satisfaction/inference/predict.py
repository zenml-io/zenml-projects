import numpy as np 
import pandas as pd 
import joblib 

import mlflow

def predict(model, payment_sequential,
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
product_width_cm):
    """
    Predict the customer satisfaction score for the test set.
    """
    input_list = [[payment_sequential,payment_installments,payment_value,price,freight_value,	
product_name_lenght,
product_description_lenght,	
product_photos_qty,	
product_weight_g,	
product_length_cm,
product_height_cm,
product_width_cm] ]
    # convert list to pd dataframe
    input_df = pd.DataFrame(input_list, columns=['payment_sequential', 'payment_installments', 'payment_value', 'price', 'freight_value', 'product_name_lenght', 'product_description_lenght', 'product_photos_qty', 'product_weight_g', 'product_length_cm', 'product_height_cm', 'product_width_cm']) 
    y_pred = model.predict(pd.DataFrame(input_df))
    return y_pred 
