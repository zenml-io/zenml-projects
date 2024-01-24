import gradio as gr
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from zenml.client import Client
import os 

ZENML_STORE_API_KEY = os.getenv("ZENML_STORE_API_KEY", None)
ZENML_STORE_URL = os.getenv("ZENML_STORE_URL", None)

if ZENML_STORE_API_KEY:
    # Use os.process to call zenml connect --url ZENML_STORE_URL --api-key ZENML_STORE_API_KEY
    os.system(f"zenml connect --url {ZENML_STORE_URL} --api-key {ZENML_STORE_API_KEY}")

client = Client()
zenml_model = client.get_model_version("breast_cancer_classifier", "production")
preprocess_pipeline = zenml_model.get_artifact("preprocess_pipeline").load()

# Load the model
clf = zenml_model.get_artifact("model").load()

# Load dataset to get feature names
data = load_breast_cancer()
feature_names = data.feature_names

def classify(*input_features):
    # Convert the input features to pandas DataFrame
    input_features = np.array(input_features).reshape(1, -1)
    input_df = pd.DataFrame(input_features, columns=feature_names)

    # Pre-process the DataFrame
    input_df["target"] = pd.Series([1] * input_df.shape[0])
    input_df = preprocess_pipeline.transform(input_df)
    input_df.drop(columns=["target"], inplace=True)

    # Make a prediction
    prediction_proba = clf.predict_proba(input_df)[0]

    # Map predicted class probabilities
    classes = data.target_names
    return {classes[idx]: prob for idx, prob in enumerate(prediction_proba)}

# Define a list of Number inputs for each feature
input_components = [gr.Number(label=feature_name, default=0) for feature_name in feature_names]

# Define the Gradio interface
iface = gr.Interface(
    fn=classify,
    inputs=input_components,
    outputs=gr.Label(num_top_classes=2),
    title="Breast Cancer Classifier",
    description="Enter the required measurements to predict the classification for breast cancer."
)

# Launch the Gradio app
iface.launch()