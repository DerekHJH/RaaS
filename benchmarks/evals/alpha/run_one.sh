
dataset="math500"
model="Qwen/Qwen2.5-Math-7B-Instruct"
all_approaches=("raas-64-0.005" "raas-128-0.005" "raas-256-0.005" "raas-512-0.005" "raas-1024-0.005" "raas-64-0.01" "raas-128-0.01" "raas-256-0.01" "raas-512-0.01" "raas-1024-0.01" "raas-64-0.02" "raas-128-0.02" "raas-256-0.02" "raas-512-0.02" "raas-1024-0.02" "raas-64-0.05" "raas-128-0.05" "raas-256-0.05" "raas-512-0.05" "raas-1024-0.05")

# Take the arguments from the command line
if [ $# -eq 0 ]; then
    echo "No arguments provided. Using default values."
elif [ $# -eq 1 ]; then
    dataset=$1
elif [ $# -eq 2 ]; then
    dataset=$1
    model=$2
else
    echo "Too many arguments provided. Usage: $0 [dataset] [model]"
    exit 1
fi


for approach in ${all_approaches[@]}; do
    command="python3 main.py --dataset ${dataset} --model ${model} --approach ${approach}"
    echo "Running command: ${command}"
    ${command}
done
