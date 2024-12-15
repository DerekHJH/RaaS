# Configure the following three variables as needed
dataset="math500"
model="peiyi9979/mistral-7b-sft"
approach="full"

command="python3 main.py --dataset ${dataset} --model ${model} --approach ${approach}"
echo "Running command: ${command}"
${command}

