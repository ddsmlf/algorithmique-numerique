#!/bin/bash

export PYTHONPATH=$(pwd):$(pwd)/tests

for test_file in tests/*_test.py; do
    test_module=$(basename "$test_file" .py)
    python -m unittest "tests.$test_module"
done