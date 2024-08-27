import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing_extensions import Annotated, Tuple

from zenml import pipeline, step
from zenml.config import DockerSettings


@step
def ingest_data():
    # Load the Paris Airbnb dataset
    data = pd.read_csv(
        "http://data.insideairbnb.com/france/ile-de-france/paris/2023-12-12/data/listings.csv.gz"
    )
    return data


from typing import List

@step
def preprocess_data(
    data: pd.DataFrame,
) -> Tuple[
    Annotated[np.ndarray, "paris_airbnb_X_train"],
    Annotated[np.ndarray, "paris_airbnb_X_test"],
    Annotated[np.ndarray, "paris_airbnb_y_train"],
    Annotated[np.ndarray, "paris_airbnb_y_test"],
    Annotated[StandardScaler, "paris_airbnb_scaler"],
]:
    def safe_convert_to_float(series: pd.Series) -> pd.Series:
        try:
            return pd.to_numeric(series.str.replace('$', '').str.replace(',', ''), errors='coerce')
        except AttributeError:
            return pd.to_numeric(series, errors='coerce')

    # Select relevant features
    features = [
        "accommodates",
        "bedrooms",
        "beds",
        "number_of_reviews",
        "review_scores_rating",
        "latitude",
        "longitude",
    ]
    
    # Check if these features exist in the dataframe
    existing_features = [f for f in features if f in data.columns]
    print(f"\nExisting features: {existing_features}")
    
    X = data[existing_features]
    
    # Convert all feature columns to float, handling errors
    for col in X.columns:
        X[col] = safe_convert_to_float(X[col])
    
    # Convert price to numeric, removing the dollar sign and converting to float
    if 'price' in data.columns:
        y = safe_convert_to_float(data['price'])
    else:
        raise ValueError("'price' column not found in the dataframe")

    # Handle missing values
    X = X.fillna(X.mean())
    y = y.fillna(y.mean())

    # Remove rows where y is NaN
    mask = ~np.isnan(y)
    X = X[mask]
    y = y[mask]

    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train.values, y_test.values, scaler

@step
def train_model(
    X_train: np.ndarray, y_train: np.ndarray
) -> Annotated[RandomForestRegressor, "paris_airbnb_model"]:
    # Remove any rows with NaN values
    mask = ~np.isnan(y_train)
    X_train_clean = X_train[mask]
    y_train_clean = y_train[mask]

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train_clean, y_train_clean)
    return model


@step
def evaluate_model(
    model: RandomForestRegressor, X_test: np.ndarray, y_test: np.ndarray
) -> Annotated[float, "paris_airbnb_mse"]:
    # Remove any rows with NaN values
    mask = ~np.isnan(y_test)
    X_test_clean = X_test[mask]
    y_test_clean = y_test[mask]

    predictions = model.predict(X_test_clean)
    mse = mean_squared_error(y_test_clean, predictions)
    return mse


@step
def generate_description(
    model: RandomForestRegressor, X_test: np.ndarray, y_test: np.ndarray, scaler: StandardScaler
) -> Annotated[str, "paris_airbnb_description"]:
    tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")
    lm_model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-v0.1")

    # Generate a description for a random listing
    random_index = np.random.randint(len(X_test))
    listing_features = X_test[random_index]
    true_price = y_test[random_index]
    predicted_price = model.predict([listing_features])[0]

    # Inverse transform to get original feature values
    original_features = scaler.inverse_transform([listing_features])[0]

    prompt = f"""
    Describe a charming Parisian Airbnb listing with the following features:
    - Accommodates: {int(original_features[0])} guests
    - Bedrooms: {int(original_features[1])}
    - Beds: {int(original_features[2])}
    - Number of reviews: {int(original_features[3])}
    - Review score: {original_features[4]:.1f}/5
    - Location: Latitude {original_features[5]:.4f}, Longitude {original_features[6]:.4f}
    - Actual price: €{true_price:.2f} per night
    - Predicted price by our model: €{predicted_price:.2f} per night

    Make it sound whimsical and charming, like a Parisian host with a poetic flair.
    """

    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = lm_model.generate(**inputs, max_new_tokens=200)
    description = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return description

@pipeline(enable_cache=True)
def paris_airbnb_pipeline():
    data = ingest_data()
    X_train, X_test, y_train, y_test, scaler = preprocess_data(data)
    model = train_model(X_train, y_train)
    mse = evaluate_model(model, X_test, y_test)
    description = generate_description(model, X_test, y_test, scaler)
    return mse, description


if __name__ == "__main__":
    paris_airbnb_pipeline()
