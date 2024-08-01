from zenml import pipeline
from steps.hf_dataset_loader import load_hf_dataset

@pipeline
def generate_synthetic_data():
    load_hf_dataset()

if __name__ == "__main__":
    generate_synthetic_data()
