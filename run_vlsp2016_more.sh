
echo "## Run"
task_name="sonba"
learning_rate=3e-5
num_train_epochs=10
train_batch_size=2
bert_model="FacebookAI/xlm-roberta-large"

# bert_model="vinai/phobert-base-v2"
data_dir="tmp_data/VLSP2016"
# data_dir="data/vlsp2016"
# data_dir="data/vlsp2018"

# bert_model="vinai/phobert-base-v2"
# bert_model="cache/xlm-large"
# bert_model="cache/visobert"
cache_dir="cache"
max_seq_length=512

python train_bert_crf_EC_xlm_roberta.py \
    --do_train \
    --do_eval \
    --output_dir "./output_vlsp2016" \
    --bert_model "${bert_model}" \
    --learning_rate ${learning_rate} \
    --data_dir "${data_dir}" \
    --num_train_epochs ${num_train_epochs} \
    --train_batch_size ${train_batch_size} \
    --task_name "${task_name}" \
    --cache_dir "${cache_dir}" \
    --max_seq_length ${max_seq_length}
