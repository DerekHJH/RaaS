import subprocess
from benchmarks.eval_engines.e2e.main import Configs

def run(dataset: str, model: str, approach: str):
    command = f'python3 main.py --dataset {dataset} --model {model} --approach {approach}'
    print(f'Running command: {command}')
    process = subprocess.Popen(command, shell=True)
    process.wait()


# Run evaluations of all combinations of datasets, models, and approaches.
# for dataset in Configs.all_datasets:
#     for model in Configs.all_models:
#         for approach in Configs.all_approaches:
#             run(dataset, model, approach)

# Run a specific evaluation.
dataset = 'needle'
model = 'peiyi9979/mistral-7b-sft'
approach = 'full'
run(dataset, model, approach)