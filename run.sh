#!/bin/bash
echo "Compiling..."
nvcc src/main.cu src/kernels.cu -Iheaders -o main
if [ $? -ne 0 ]; then
    echo "Compilation failed"
    exit 1
fi

echo "Running..."
./main