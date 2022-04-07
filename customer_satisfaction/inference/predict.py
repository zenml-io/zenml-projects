import numpy as np
import pandas as pd
from sklearn.base import RegressorMixin


def predict(
    model: RegressorMixin,
    payment_sequential: int,
    payment_installments: int,
    payment_value: int,
    price: int,
    freight_value: int,
    product_name_length: int,
    product_description_length: int,
    product_photos_qty: int,
    product_weight_g: int,
    product_length_cm: int,
    product_height_cm: int,
    product_width_cm: int,
) -> np.ndarray:
    """It predicts the customer satisfaction score for the test set.

    Args:
        model: The model that will be used to predict the score
        payment_sequential: The sequential number of the payment
        payment_installments: Number of installments of the product
        payment_value: The value of the order
        price: the product's price
        freight_value: The freight value of the order
        product_name_length: The number of characters in the product name
        product_description_length: The length of the product description
        product_photos_qty: number of photos of the item
        product_weight_g: Weight of the product (in grams)
        product_length_cm: Length of the product, in centimeters
        product_height_cm: height of the product in cm
        product_width_cm: The width of the product, in centimeters

    Outputs:
        A numpy array of the predicted customer satisfaction score.
    """

    input_list = [
        [
            payment_sequential,
            payment_installments,
            payment_value,
            price,
            freight_value,
            product_name_length,
            product_description_length,
            product_photos_qty,
            product_weight_g,
            product_length_cm,
            product_height_cm,
            product_width_cm,
        ]
    ]
    input_df = pd.DataFrame(
        input_list,
        columns=[
            "payment_sequential",
            "payment_installments",
            "payment_value",
            "price",
            "freight_value",
            "product_name_length",
            "product_description_length",
            "product_photos_qty",
            "product_weight_g",
            "product_length_cm",
            "product_height_cm",
            "product_width_cm",
        ],
    )
    y_pred = model.predict(pd.DataFrame(input_df))
    return y_pred
