batch_size=2
question_mask_ratio=0.5
distillation_temp=1.0
compressor_hidden_size=4096
num_compressor_layers=4
num_compressor_encoder_layers=2
pool_window_size=4
cand_pool_window_sizes=(4 6 8 10)
min_num_documents=1
max_num_documents=5
compressor_path=/path-to-compressor

# target LLM is LongChat-13B
lm_model_name=longchat
lm_model_hidden_size=5120
lm_model_path=/path-to-longchat-13b

# # target LLM is LLaMA-2-7B
# lm_model_name=llama
# lm_model_hidden_size=4096
# lm_model_path=/path-to-llama-2-7b

data_path=/path-to-data
dev_steps=500
test_steps=500
save_steps=1000
logging_steps=100
benchmark_dev_steps=1000
benchmark_test_steps=1000
benchmark_metric=accuracy

output_dir=/path-to-save
mkdir -p ${output_dir}

accelerate launch --config_file config/bf16.yaml \
    src/train.py \
    --data_path $data_path \
    --compressor_path $compressor_path \
    --lm_model_name $lm_model_name \
    --lm_model_path $lm_model_path \
    --output_dir $output_dir \
    --question_mask_ratio $question_mask_ratio \
    --compressor_hidden_size $compressor_hidden_size \
    --lm_model_hidden_size $lm_model_hidden_size \
    --num_compressor_layers $num_compressor_layers \
    --num_compressor_encoder_layers $num_compressor_encoder_layers \
    --max_num_documents $max_num_documents \
    --min_num_documents $min_num_documents \
    --pool_window_size $pool_window_size \
    --train_batch_size $batch_size \
    --eval_batch_size $batch_size \
    --random_num_documents \
    --dev_steps $dev_steps \
    --test_steps $test_steps \
    --save_steps $save_steps \
    --logging_steps $logging_steps \
    --benchmark_dev_steps $benchmark_dev_steps \
    --benchmark_test_steps $benchmark_test_steps \
    --benchmark_metric $benchmark_metric \
    --random_pool_window_size \
    --cand_pool_window_sizes ${cand_pool_window_sizes[@]} \
    | tee ${output_dir}/train.log