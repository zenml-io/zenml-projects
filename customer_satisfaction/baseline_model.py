from random import random
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import os
import joblib


class DataClass:
    def __init__(self) -> None:
        self.data_files = os.listdir("./data")
        self.data_files = [
            file for file in self.data_files if file.endswith(".csv")
        ]
        self.data_files = ["./data/" + file for file in self.data_files]

    def read_data(self) -> pd.DataFrame:
        print(self.data_files)
        data = pd.read_csv("./data/olist_customers_dataset.csv")
        geo_data = pd.read_csv("./data/olist_geolocation_dataset.csv")
        order_itemdata = pd.read_csv("./data/olist_order_items_dataset.csv")
        pay_data = pd.read_csv("./data/olist_order_payments_dataset.csv")
        rev_data = pd.read_csv("./data/olist_order_reviews_dataset.csv")
        orders = pd.read_csv("./data/olist_orders_dataset.csv")
        order_prddata = pd.read_csv("./data/olist_products_dataset.csv")
        order_selldata = pd.read_csv("./data/olist_sellers_dataset.csv")
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

    def analyze_data(self, df: pd.DataFrame) -> None:
        # print columns which are time column
        # remove order_approved_at, order_delivered_carrier_date, order_delivered_customer_date, order_estimated_delivery_date, order_purchase_timestamp
        df = df.drop(
            [
                "order_approved_at",
                "order_delivered_carrier_date",
                "order_delivered_customer_date",
                "order_estimated_delivery_date",
                "order_purchase_timestamp",
            ],
            axis=1,
        )
        df["product_weight_g"].fillna(
            df["product_weight_g"].median(), inplace=True
        )
        df["product_length_cm"].fillna(
            df["product_length_cm"].median(), inplace=True
        )
        df["product_height_cm"].fillna(
            df["product_height_cm"].median(), inplace=True
        )
        df["product_width_cm"].fillna(
            df["product_width_cm"].median(), inplace=True
        )
        # write "No review" in review_comment_message column
        df["review_comment_message"].fillna("No review", inplace=True)

        return df

    def preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        #  only keep columns which are float or int type
        df = df.select_dtypes(include=[np.number])
        cols_to_drop = ["customer_zip_code_prefix", "order_item_id"]
        df = df.drop(cols_to_drop, axis=1)
        return df


class ModelBuilding:
    def __init__(self, df: pd.DataFrame) -> None:
        self.df = df
        self.X = df.drop("review_score", axis=1)
        self.y = df["review_score"]
        (
            self.X_train,
            self.X_test,
            self.y_train,
            self.y_test,
        ) = train_test_split(self.X, self.y, test_size=0.2, random_state=0)

    def build_model(self) -> None:
        random_forest = RandomForestRegressor(n_estimators=100, random_state=0)
        random_forest.fit(self.X_train, self.y_train)
        return random_forest

    def evaluate_model(self, model) -> None:
        pred = model.predict(self.X_test)
        mse = mean_squared_error(self.y_test, pred)
        print("Mean squared error: %.2f" % mse)
        rmse = np.sqrt(mse)
        print("Root mean squared error: %.2f" % rmse)

    def save_model(self, model, name_of_model) -> None:
        joblib.dump(model, name_of_model)


if __name__ == "__main__":
    # df = DataIngestion().read_data()
    # df.to_csv("./data/olist_customers_dataset.csv", index=False)
    df = pd.read_csv("./data/olist_customers_dataset.csv")
    df = DataClass().analyze_data(df)
    df = DataClass().preprocess_data(df)

    model = ModelBuilding(df)
    rf_model = model.build_model()
    model.evaluate_model(rf_model)
    model.save_model(rf_model, "rf_model.pkl")
