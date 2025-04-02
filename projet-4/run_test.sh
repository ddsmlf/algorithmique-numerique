#!/bin/bash

export PYTHONPATH=$(pwd):$(pwd)/tests

RED='\033[0;31m'
GREEN='\033[0;32m'
NC='\033[0m' # No Color

for test_file in tests/*_test.py; do
    test_module=$(basename "$test_file" .py)
    test_name=$(basename "$test_file" _test.py)
    if python3 -m unittest "tests.$test_module"; then
        echo -e "${GREEN}Test passed: $test_name${NC}"
    else
        echo -e "${RED}Test failed: $test_name${NC}"
    fi
done