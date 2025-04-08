import gradio as gr
import pandas as pd
import xgboost as xgb
from zenml.client import Client

# Initialize ZenML client
client = Client()

# Fetch the latest production model
zenml_model = client.get_model_version("ecb_interest_rate_model", "production")
model: xgb.Booster = zenml_model.get_artifact("xgb_model").load()


def predict(deposit_rate, marginal_rate):
    # Calculate the features as they were during training
    augmented_rate = (
        deposit_rate * 2
    )  # This is an assumption; adjust if it was calculated differently
    rate_diff = marginal_rate - deposit_rate

    # Prepare input data
    input_data = pd.DataFrame(
        {"augmented_rate": [augmented_rate], "rate_diff": [rate_diff]}
    )

    # Make prediction
    dmatrix = xgb.DMatrix(input_data)
    prediction = model.predict(dmatrix)[0]

    return f"Predicted Main Refinancing Rate: {prediction:.4f}"


# Create Gradio interface
iface = gr.Interface(
    fn=predict,
    inputs=[
        gr.Slider(
            minimum=0, maximum=10, step=0.01, label="Deposit Facility Rate"
        ),
        gr.Slider(
            minimum=0,
            maximum=10,
            step=0.01,
            label="Marginal Lending Facility Rate",
        ),
    ],
    outputs="text",
    title="ECB Main Refinancing Rate Predictor",
    description="Enter the Deposit Facility Rate and Marginal Lending Facility Rate to predict the Main Refinancing Rate.",
)

# Launch the app
iface.launch()
