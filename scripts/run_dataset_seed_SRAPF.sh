methods=("finetune")

data_sources=("fewshot+retrieved")

folder="SRAPF"

cls_inits=("openai") # OpenAI 80 prompts average


# shot_values=(4 8 16)
shot_values=(16)


batch_size=64
epochs=10
lr_backbone=1e-6
lr_classifier=1e-3
wd=1e-2
warmup_lr=1e-8
warmup_iter=18


log_mode="both"


#------------------------------
# DO NOT MODIFY BELOW THIS LINE !!!
#------------------------------

# Check if command-line arguments were provided
if [ "$#" -ge 3 ]; then
    datasets=("$1")  # Use the provided command-line argument for the dataset
    seeds=("$2")  # Use the provided command-line argument for the seed
    ft_topk_blks="$3"  # Use the provided command-line argument for topk blocks to finetune
else
    echo "Usage: $0 <dataset> <seed> <ft_topk_blks>"
fi

# add _topk to folder name if ft_topk_blks is not -1
if [ "$ft_topk_blks" -ne -1 ]; then
    folder="${folder}_top${ft_topk_blks}"
else
    folder="${folder}_all"
fi

output_folder="output/$folder"
if [ ! -d "$output_folder" ]; then
    mkdir -p "$output_folder"
fi


# Loop through all combinations and run the script
for dataset in "${datasets[@]}"; do
    for method in "${methods[@]}"; do
        for data_source in "${data_sources[@]}"; do
            for shots in "${shot_values[@]}"; do
                for init in "${cls_inits[@]}"; do
                    for seed in "${seeds[@]}"; do
                        echo "Running: $dataset $method $data_source $init $shots $seed $retrieval_split"

                        # Run the script and capture the output
                        output=$(python main.py --dataset "$dataset" --method "$method" --data_source "$data_source"  \
                        --cls_init "$init" --num_shots "$shots" --data_seed "$seed" --epochs "$epochs" --bsz "$batch_size" \
                        --log_mode "$log_mode" --lr_backbone "$lr_backbone" --lr_classifier "$lr_classifier" --wd "$wd" \
                        --warmup_lr "$warmup_lr" --warmup_iter "$warmup_iter" --folder "$output_folder" --early_stop \
                        --ft_topk_blks "$ft_topk_blks" --add_ap_stage2 --eps_stage2 0.01 \
                        )

                        # Print the output to the console
                        echo "$output"

                    done
                done
            done
        done
    done
done
