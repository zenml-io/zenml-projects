# This is an example feature definition file

from datetime import timedelta
from tokenize import String

from feast import Entity, FeatureView, Field, FileSource, ValueType
from feast.types import Float32, Int64
from numpy import float64

# Read data from parquet files. Parquet is convenient for local development mode. For
# production, you can use your favorite DWH, such as BigQuery. See Feast documentation
# for more info.
customer_churn = FileSource(
    path="customer-churn/feature_repo/data/customer-churn-data.parquet",
    timestamp_field="event_timestamp",
    created_timestamp_column="created",
)

# Define an entity for the driver. You can think of entity as a primary key used to
# fetch features.
driver = Entity(
    name="customerID",
    value_type=ValueType.BYTES,
    description="customerID",
)

# Our parquet files contain sample data that includes a driver_id column, timestamps and
# three feature column. Here we define a Feature View that will allow us to serve this
# data to our model online.
customer_churn = FeatureView(
    name="customer-churn",
    entities=["customerID"],
    ttl=timedelta(days=1),
    schema=[
        Field(name="gender", dtype=ValueType.UNKNOWN),
        Field(name="SeniorCitizen", dtype=ValueType.INT64),
        Field(name="Partner", dtype=ValueType.STRING),
        Field(name="Dependents", dtype=ValueType.STRING),
        Field(name="tenure", dtype=ValueType.INT64),
        Field(name="PhoneService", dtype=ValueType.STRING),
        Field(name="MultipleLines", dtype=ValueType.STRING),
        Field(name="InternetService", dtype=ValueType.STRING),
        Field(name="OnlineSecurity", dtype=ValueType.STRING),
        Field(name="OnlineBackup", dtype=ValueType.STRING),
        Field(name="DeviceProtection", dtype=ValueType.STRING),
        Field(name="TechSupport", dtype=ValueType.STRING),
        Field(name="StreamingTV", dtype=ValueType.STRING),
        Field(name="StreamingMovies", dtype=ValueType.STRING),
        Field(name="Contract", dtype=ValueType.STRING),
        Field(name="PaperlessBilling", dtype=ValueType.STRING),
        Field(name="PaymentMethod", dtype=ValueType.STRING),
        Field(name="MonthlyCharges", dtype=ValueType.FLOAT),
        Field(name="TotalCharges", dtype=ValueType.STRING),
        Field(name="Churn", dtype=ValueType.STRING),
    ],
    online=True,
    batch_source=customer_churn,
    tags={},
)
