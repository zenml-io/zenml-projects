#!/usr/bin/env bash
set -e
set -x
set -o pipefail

LINT_FILES=${1:-"."}

export ZENML_DEBUG=1
export ZENML_ANALYTICS_OPT_IN=false

ruff $LINT_FILES
# TODO: Fix docstrings in tests and examples and remove the `--extend-ignore D` flag
ruff $LINT_FILES --extend-ignore D

# autoflake replacement: checks for unused imports and variables
ruff $LINT_FILES --select F401,F841 --exclude "__init__.py" --isolated

black $SRC  --check

# check type annotations
mypy $LINT_FILES
