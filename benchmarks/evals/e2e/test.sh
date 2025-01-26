
dataset="math500"
model="peiyi9979/mistral-7b-sft"
approach="full_optimized"

command="python3 main.py --dataset ${dataset} --model ${model} --approach ${approach}"
echo "Running command: ${command}"
${command}

