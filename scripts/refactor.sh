#!/bin/bash

# Formats all tracked Python files with ruff format (configured in pyproject.toml); it only changes the layout.

ruff format $(git ls-files '*.py')
