dataset="math500"
model="Qwen/Qwen2.5-Math-7B-Instruct"
approach="raas-1024-0.01"

command="python3 main.py --dataset ${dataset} --model ${model} --approach ${approach}"
echo "Running command: ${command}"
${command}

