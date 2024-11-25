import subprocess
from benchmarks.eval_engines.e2e.main import Configs

# Configure the following three variables as needed
dataset = 'needle'
model = 'peiyi9979/mistral-7b-sft'
approach = 'full'


command = f'python3 main.py --dataset {dataset} --model {model} --approach {approach}'
print(f'Running command: {command}')
process = subprocess.Popen(command, shell=True)
process.wait()