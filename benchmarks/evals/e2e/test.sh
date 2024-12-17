
dataset="aime"
model="AIDC-AI/Marco-o1"
approach="full"

command="python3 main.py --dataset ${dataset} --model ${model} --approach ${approach}"
echo "Running command: ${command}"
${command}

