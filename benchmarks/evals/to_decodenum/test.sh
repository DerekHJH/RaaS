dataset="gsm8k"
model="peiyi9979/mistral-7b-sft"
approach="raas_optimized-64"

command="python3 main.py --dataset ${dataset} --model ${model} --approach ${approach}"
echo "Running command: ${command}"
${command}

