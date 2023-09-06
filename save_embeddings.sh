output_dir="/home/vk352/IncDSI/test_output/saved_embeddings/"
model_path="/home/vk352/IncDSI/test_output/initial_model/base_model_epoch20"

for doc_split in 'old' 'new' 'tune'
do
    for split in 'train' 'val' 'test' 'gen'
    do
        train_cmd="
        python save_embeddings.py --output_dir=$output_dir \
        --model_name='bert-base-uncased' --split=$split \
        --initialize_model=$model_path --dataset 'nq320k' --doc_split=$doc_split"

        echo $train_cmd
        eval $train_cmd
    done
done