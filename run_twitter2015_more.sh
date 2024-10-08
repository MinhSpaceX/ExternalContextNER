
echo "## Run"
task_name="sonba"
learning_rate=3e-5
num_train_epochs=10
train_batch_size=8
bert_model="google-bert/bert-base-cased"

# bert_model="vinai/phobert-base-v2"
data_dir="tmp_data/twitter2015-txt"
# data_dir="data/vlsp2016"
# data_dir="data/vlsp2018"

# bert_model="vinai/phobert-base-v2"
# bert_model="cache/xlm-large"
# bert_model="cache/visobert"
cache_dir="cache"
max_seq_length=512

python train_bert_crf_EC_new.py \
    --do_train \
    --do_eval \
    --output_dir "./output_twitter2015txt" \
    --bert_model "${bert_model}" \
    --learning_rate ${learning_rate} \
    --data_dir "${data_dir}" \
    --num_train_epochs ${num_train_epochs} \
    --train_batch_size ${train_batch_size} \
    --task_name "${task_name}" \
    --cache_dir "${cache_dir}" \
    --max_seq_length ${max_seq_length}