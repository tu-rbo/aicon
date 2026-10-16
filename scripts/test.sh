#!/bin/bash

# Linters (as in the CI) and tests with coverage; updates the badges in docs/badges. See ./scripts/test.sh --help.

RED='\033[1;31m'
YELLOW='\033[1;33m'
# No colour
NC='\033[0m'

clear
# Activate the virtual environment created by `uv sync` if there is one (pixi activates its own)
if [ -f .venv/bin/activate ]; then
    source .venv/bin/activate
fi

function help() {
    echo "Script for enforce coding quality."
    echo
    echo "SYNOPSIS"
    echo "  ./test.sh [OPTION]"
    echo
    echo "OPTIONS"
    echo "                  Without option, run both coding stype checks and pytest"
    echo "  -h, --help      Display help"
    echo "  -s, --style     Run only coding style check with Pylint and Flake8"
    echo "  -p, --pytest    Run only unit and integration tests with Pytest"
    exit
}

function error-invalid-option() {
    echo -e "${RED}ERROR: Invalid option${NC}"
    echo "Check './test.sh --help' for valid ones."
    exit
}

# Enforce coding style with pylint and flake8
function test_coding_style() {
    echo -e "${YELLOW}Testing coding style ...${NC}"
    ruff format --check $(git ls-files '*.py')
    pylint -j=0 $(git ls-files '*.py')
    flake8 $(git ls-files '*.py')
}

# Run pytest, coverage check, save reports and badges
function run_pytest_and_coverage() {
    # Run pytest and coverage check
    echo -e "${YELLOW}Run pytest and coverage check ...${NC}"
    # pytest --durations=0  # To check the duration of slow tests
    coverage run --source=aicon_base -m pytest --durations=10 --maxfail=2 --junitxml=reports/junit/junit.xml &&
        coverage report -m
    coverage xml -o reports/coverage/coverage.xml

    # Save test badges
    echo -e "${YELLOW}Generating test and coverage badges ...${NC}"
    genbadge tests -o docs/badges/tests-badge.svg
    genbadge coverage -o docs/badges/coverage-badge.svg
}

########################################################
# MAIN
########################################################
while [ "$1" != "" ]; do
    case $1 in
    -h | --help) help ;;
    -s | --style) test_coding_style ;;
    -p | --pytest) run_pytest_and_coverage ;;
    *) error-invalid-option ;;
    esac
    shift
    exit
done

echo -e "${YELLOW}WARNING: test.sh missing operand${NC}"
echo "Running all scripts by default."
echo "try './test.sh --help' for more information."
test_coding_style
run_pytest_and_coverage
