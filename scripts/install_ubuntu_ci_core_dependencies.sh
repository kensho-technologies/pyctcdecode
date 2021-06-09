#!/usr/bin/env bash
# Copyright 2020-present Kensho Technologies, LLC.

# Treat undefined variables and non-zero exits in pipes as errors.
set -uo pipefail

# Ensure that the "**" glob operator is applied recursively.
# Make globs that do not match return null values.
shopt -s globstar nullglob

# Break on first error.
set -e

# Ensure pip, setuptools, and pipenv are latest available versions.
python -m pip install --upgrade pip
