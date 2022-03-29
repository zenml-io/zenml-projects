import os

import numpy as np
import pandas as pd


class IngestData:
    def __init__(self) -> None:
        pass

    def read_data(self) -> pd.DataFrame:
        data = pd.read_csv("./data/olist_customers_dataset.csv")
        order_itemdata = pd.read_csv("./data/olist_order_items_dataset.csv")
        pay_data = pd.read_csv("./data/olist_order_payments_dataset.csv")
        rev_data = pd.read_csv("./data/olist_order_reviews_dataset.csv")
        orders = pd.read_csv("./data/olist_orders_dataset.csv")
        order_prddata = pd.read_csv("./data/olist_products_dataset.csv")
        order_prd_catdata = pd.read_csv(
            "./data/product_category_name_translation.csv"
        )
        rev_new = rev_data.drop(
            [
                "review_comment_title",
                "review_creation_date",
                "review_id",
                "review_answer_timestamp",
            ],
            axis=1,
        )
        df = pd.merge(orders, pay_data, on="order_id")
        df = df.merge(data, on="customer_id")
        df = df.merge(order_itemdata, on="order_id")
        df = df.merge(order_prddata, on="product_id")
        df = df.merge(order_prd_catdata, on="product_category_name")
        df = df.merge(rev_new, on="order_id")
        return df

    def get_data(self) -> pd.DataFrame: 
        df = pd.read_csv("./data/olist_customers_dataset.csv") 
        return df 
    def download_data(self, link: str) -> None:
        """
        TODO: Need to Test it ( if it is working or not)
        """
        link = "https://ayushml.blob.core.windows.net/data/datacustomersat.zip"
        os.system("wget " + link)
        os.system("unzip datacustomersat.zip")
        os.system("rm datacustomersat.zip")
        os.system("mv datacustomersat.csv ./data/")
        os.system("mv datacustomersat.csv.csv ./data/")

    def get_data_for_test(self) -> pd.DataFrame:
        df = pd.read_csv("./data/olist_customers_dataset.csv")
        # take sample from the data 
        df = df.sample(n=100)
        return df
        