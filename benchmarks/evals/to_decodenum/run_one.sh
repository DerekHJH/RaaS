
dataset="gsm8k"
model="peiyi9979/mistral-7b-sft"
# all_approaches=("full" "sink-64" "sink-128" "sink-256" "sink-512" "sink-1024" "quest-64" "quest-128" "quest-256" "quest-512" "quest-1024")
# all_approaches=("raas-64" "raas-128" "raas-256" "raas-512" "raas-1024")
all_approaches=(
    "raas_optimized-64"
    "raas_optimized-128"
    "raas_optimized-256"
    "raas_optimized-512"
    "raas_optimized-1024"
    "quest_optimized-64"
    "quest_optimized-128"
    "quest_optimized-256"
    "quest_optimized-512"
    "quest_optimized-1024"
    "full_optimized"
    # "sink_optimized-64"
    # "sink_optimized-128"
    # "sink_optimized-256"
    # "sink_optimized-512"
    # "sink_optimized-1024"
)

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
