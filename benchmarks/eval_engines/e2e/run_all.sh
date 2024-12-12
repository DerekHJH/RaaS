#!/bin/bash

for dataset in "math500"; do
    for model in "peiyi9979/mistral-7b-sft" "Qwen/Qwen2.5-Math-7B-Instruct"; do
        for approach in "full" "streamingllm"; do
            command="python3 main.py --dataset ${dataset} --model ${model} --approach ${approach}"
            echo "Running command: ${command}"
            ${command}
        done
    done
done