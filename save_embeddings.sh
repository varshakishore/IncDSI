model_path="$1"
output_dir="$2"
dataset="$3"

for doc_split in 'old' 'new' 'tune'
do
    for split in 'train' 'val' 'test' 'gen'
    do
        train_cmd="
        python save_embeddings.py --output_dir=$output_dir \
        --model_name='bert-base-uncased' --split=$split \
        --initialize_model=$model_path --dataset=$dataset --doc_split=$doc_split"

        echo $train_cmd
        eval $train_cmd
    done
done