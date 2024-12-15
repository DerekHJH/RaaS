
dataset="aime"
model="peiyi9979/mistral-7b-sft"
approache="quest-1024"

command="python3 main.py --dataset ${dataset} --model ${model} --approach ${approach}"
echo "Running command: ${command}"
${command}

