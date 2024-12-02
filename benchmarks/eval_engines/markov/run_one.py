import subprocess

# These variables are ususally fixed for Markov Property testing
dataset = "math500"
model = "peiyi9979/mistral-7b-sft"
approach = "full"


command = f"python3 main.py --dataset {dataset} --model {model} --approach {approach}"
print(f"Running command: {command}")
process = subprocess.Popen(command, shell=True)
process.wait()
