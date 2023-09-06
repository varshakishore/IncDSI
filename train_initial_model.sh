# hyper parameters
batch_size=128
learning_rate=0.00001
train_epochs=20
logging_step=200
base_data_dir="$1"
output_dir="$2"
wandb_name="initial_model"

train_cmd="
python  train_initial_model.py \
--batch_size=$batch_size  --output_dir=$output_dir --logging_step=$logging_step \
--learning_rate $learning_rate  --train_epochs $train_epochs --model_name='bert-base-uncased' \
--base_data_dir=$base_data_dir"

echo $train_cmd
eval $train_cmd

mkdir -p $output_dir
echo "copy current script to model directory to:"
echo $output_dir
cp $0 $output_dir
