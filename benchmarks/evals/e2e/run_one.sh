
dataset="aime"
model="peiyi9979/mistral-7b-sft"
# all_approaches=("full" "sink-64" "sink-128" "sink-256" "sink-512" "sink-1024")
all_approaches=("quest-64" "quest-128" "quest-256" "quest-512" "quest-1024")

# Take the arguments from the command line
if [ $# -eq 2 ]; then
    dataset=$1
    model=$2
fi


for approach in ${all_approaches[@]}; do
    command="python3 main.py --dataset ${dataset} --model ${model} --approach ${approach}"
    echo "Running command: ${command}"
    ${command}
done

