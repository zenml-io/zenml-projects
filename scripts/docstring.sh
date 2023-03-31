#!/usr/bin/env bash
set -e
set -x

DOCSTRING_SRC=${1:-"."}

export ZENML_DEBUG=1
export ZENML_ANALYTICS_OPT_IN=false

darglint -v 2 $DOCSTRING_SRC
