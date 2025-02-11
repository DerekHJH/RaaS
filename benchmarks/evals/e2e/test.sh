dataset="aime"
model="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
approach="full"

command="python3 main.py --dataset ${dataset} --model ${model} --approach ${approach}"
echo "Running command: ${command}"
${command}

