
```commandline
zenml integration install label_studio torch gcp
pip install -r requirements.txt
pip uninstall wandb
```

## Maybe

```shell
zenml integration install mlflow -y
```

# Env variables

```bash
export DATA_UPLOAD_MAX_NUMBER_FILES=1000000
export WANDB_DISABLED=True
```

# Fiftyone plugins

```shell
fiftyone plugins download https://github.com/jacobmarks/fiftyone-albumentations-plugin

fiftyone plugins download https://github.com/voxel51/fiftyone-plugins --plugin-names @voxel51/annotation
```
