#!/usr/bin/env bash
set -e
set -x

SRC=${1:-"run*.py steps/ pipelines/"}

export ZENML_DEBUG=1
export ZENML_ANALYTICS_OPT_IN=false
flake8 $SRC
autoflake --remove-all-unused-imports --recursive --remove-unused-variables --in-place $SRC --exclude=__init__.py --check
isort $SRC scripts --check-only
black $SRC --check
interrogate $SRC -c pyproject.toml
