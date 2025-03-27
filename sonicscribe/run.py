from zenml import step, pipeline


@step
def upload_audio_file(folder_path: str):
    print("Hello from sonicscribe!")


@pipeline
def my_pipeline():
    upload_audio_file(folder_path="audio_files")


if __name__ == "__main__":
    my_pipeline()
