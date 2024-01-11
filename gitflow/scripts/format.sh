#!/bin/sh -e
set -x

SRC=${1:-"run*.py steps/ pipelines/"}

export ZENML_DEBUG=1
export ZENML_ANALYTICS_OPT_IN=false
autoflake --remove-all-unused-imports --recursive --remove-unused-variables --in-place $SRC --exclude=__init__.py
isort $SRC
black $SRC
