from zenml import step, pipeline

@step
def load_data():
    pass

@step(step_operator="k8s_step_operator")
def train_model():
    pass

@step(step_operator="k8s_step_operator")
def batch_inference():
    pass


@pipeline
def dreambooth_pipeline():
    data = load_data()
    train_model(data)
    batch_inference()
