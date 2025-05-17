#!/usr/bin/env bash
set -x

export ZENML_DEBUG=1
export ZENML_ANALYTICS_OPT_IN=false

# Initialize default source directories
default_src="."
# Initialize SRC as an empty string
SRC=""

# If no source directories were provided, use the default
if [ -z "$SRC" ]; then
    SRC="$default_src"
fi

export ZENML_DEBUG=1
export ZENML_ANALYTICS_OPT_IN=false

# autoflake replacement: removes unused imports and variables
ruff check $SRC --select F401,F841 --fix --exclude "__init__.py" --exclude "llm-finetuning/" --exclude "sign-language-detection-yolov5/model.py" --exclude "*.ipynb" --isolated

# sorts imports
ruff check $SRC --exclude "llm-finetuning/" --exclude "sign-language-detection-yolov5/model.py" --select I --fix --ignore D
ruff format $SRC --exclude "sign-language-detection-yolov5/model.py" --exclude "llm-finetuning/"

