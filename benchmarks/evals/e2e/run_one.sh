# Configure the following three variables as needed
dataset="aime"
# model="AIDC-AI/Marco-o1"
# model="peiyi9979/mistral-7b-sft"
model="Qwen/Qwen2.5-Math-7B-Instruct"
approach="full"

command="python3 main.py --dataset ${dataset} --model ${model} --approach ${approach}"
echo "Running command: ${command}"
${command}

