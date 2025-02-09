dataset="aime"
model="peiyi9979/mistral-7b-sft"
approach="h2o-84"

command="python3 main.py --dataset ${dataset} --model ${model} --approach ${approach}"
echo "Running command: ${command}"
${command}

