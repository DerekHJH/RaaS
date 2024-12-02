import subprocess

# Configure the following three variables as needed
dataset = "math500"
model = "peiyi9979/mistral-7b-sft"
approach = "full"
num_tot_data = 1


command = (
    f"python3 main.py --dataset {dataset} --model {model}"
    f" --approach {approach} --num_tot_data {num_tot_data}"
)
print(f"Running command: {command}")
process = subprocess.Popen(command, shell=True)
process.wait()
