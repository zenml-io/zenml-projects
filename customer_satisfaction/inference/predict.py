import numpy as np 
import pandas as pd 
from lightgbm import LGBMRegressor

def predict(model: LGBMRegressor,
            payment_sequential: int,
            payment_installments: int,
            payment_value: int,	
            price: int,	
            freight_value: int,	
            product_name_lenght: int,
            product_description_lenght: int,	
            product_photos_qty: int,	
            product_weight_g: int,	
            product_length_cm: int,
            product_height_cm: int,
            product_width_cm: int 
) -> np.ndarray:
    """
    Predict the customer satisfaction score for the test set.
    """
    input_list = [[payment_sequential,payment_installments,payment_value,price,freight_value,product_name_lenght,product_description_lenght,product_photos_qty,	product_weight_g,product_length_cm,product_height_cm,product_width_cm] ]
    input_df = pd.DataFrame(input_list, columns=['payment_sequential', 'payment_installments', 'payment_value', 'price', 'freight_value', 'product_name_lenght', 'product_description_lenght', 'product_photos_qty', 'product_weight_g', 'product_length_cm', 'product_height_cm', 'product_width_cm']) 
    y_pred = model.predict(pd.DataFrame(input_df))
    return y_pred 
