import subprocess

for dataset in ['math500']:
    for model in ['peiyi9979/mistral-7b-sft']:
        for approach in ['full', 'quest']:
            command = f'python3 main.py --dataset {dataset} --model {model} --approach {approach}'
            print(f'Running command: {command}')
            process = subprocess.Popen(command, shell=True)
            process.wait()
