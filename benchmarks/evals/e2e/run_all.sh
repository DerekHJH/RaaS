#!/bin/bash

for dataset in "gsm8k"; do
    for model in "peiyi9979/mistral-7b-sft"; do
        for approach in "full" "sink-128"; do
            command="python3 main.py --dataset ${dataset} --model ${model} --approach ${approach}"
            echo "Running command: ${command}"
            ${command}
        done
    done
done