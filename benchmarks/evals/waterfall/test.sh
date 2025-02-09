# Configure the following three variables as needed
dataset="math500"
model="Qwen/Qwen2.5-Math-7B-Instruct"
approach="h2o-256"

command="python3 main.py --dataset ${dataset} --model ${model} --approach ${approach}"
echo "Running command: ${command}"
${command}

