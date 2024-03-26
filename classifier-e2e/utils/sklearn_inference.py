import os
from io import StringIO

import joblib
import pandas as pd

"""
Deserialize fitted model
"""


def model_fn(model_dir):
    model = joblib.load(os.path.join(model_dir, "sklearn-model"))
    return model


"""
input_fn
    request_body: The body of the request sent to the model.
    request_content_type: (string) specifies the format/variable type of the request
"""


def input_fn(request_body, request_content_type):
    if request_content_type == "text/csv":
        request = pd.read_csv(StringIO(request_body))
        return request
    else:
        raise ValueError("This model only supports text/csv input")


"""
predict_fn
    input_data: returned array from input_fn above
    model (sklearn model) returned model loaded from model_fn above
"""


def predict_fn(input_data, model):
    return model.predict(input_data)


"""
output_fn
    prediction: the returned value from predict_fn above
    content_type: the content type the endpoint expects to be returned. Ex: JSON, string

"""


def output_fn(prediction, content_type):
    return pd.Series(prediction).to_csv(index=False, header=False)
