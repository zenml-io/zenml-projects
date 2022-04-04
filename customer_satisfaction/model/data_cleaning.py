import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


class DataCleaning: 
    ''' 
    Data Cleaning class which preprocesses the data and divides it into train and test data. 
    '''
    def __init__(self, data) -> None:
        self.df = data

    def preprocess_data(self) -> pd.DataFrame:
        """
        Removes columns which are not required, fills missing values with median average values, and converts the data type to float.
        """
        self.df = self.df.drop(
            [
                "order_approved_at",
                "order_delivered_carrier_date",
                "order_delivered_customer_date",
                "order_estimated_delivery_date",
                "order_purchase_timestamp",
            ],
            axis=1,
        )
        self.df["product_weight_g"].fillna(
            self.df["product_weight_g"].median(), inplace=True
        )
        self.df["product_length_cm"].fillna(
            self.df["product_length_cm"].median(), inplace=True
        )
        self.df["product_height_cm"].fillna(
            self.df["product_height_cm"].median(), inplace=True
        )
        self.df["product_width_cm"].fillna(
            self.df["product_width_cm"].median(), inplace=True
        )
        # write "No review" in review_comment_message column
        self.df["review_comment_message"].fillna("No review", inplace=True)

        self.df = self.df.select_dtypes(include=[np.number])
        cols_to_drop = [
            "customer_zip_code_prefix",
            "order_item_id",
        ]
        self.df = self.df.drop(cols_to_drop, axis=1)

        return self.df

    def divide_data(self, df: pd.DataFrame) -> pd.DataFrame: 
        ''' 
        It divides the data into train and test data.  
        '''
        X = df.drop("review_score", axis=1)
        y = df["review_score"]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        return X_train, X_test, y_train, y_test

