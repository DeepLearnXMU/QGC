batch_size=2
question_mask_ratio=0.5
distillation_temp=1.0
compressor_hidden_size=4096
num_compressor_layers=4
num_compressor_encoder_layers=2
pool_window_size=4
cand_pool_window_sizes=(4 6 8 10)
min_num_documents=1 # 1 for NQ and TQA, 2 for HQA
max_num_documents=5
compressor_path=/path-to-llama-2-7B # used to initial compressor parameters

# target LLM is LongChat-13B
lm_model_name=longchat
lm_model_hidden_size=5120
lm_model_path=/path-to-longchat-13B

# # target LLM is LLaMA-2-7B
# lm_model_name=llama
# lm_model_hidden_size=4096
# lm_model_path=/path-to-llama-2-7B

data_path=/path-to-dataset
max_steps=20000
dev_steps=500
test_steps=500
save_steps=1000
logging_steps=100
benchmark_dev_steps=1000
benchmark_test_steps=1000

instruction_name=base # 'base' for NQ, 'short' for TQA and HQA
benchmark_metric=accuracy # NQ: accuracy; TQA: em; HQA: f1

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
    --instruction_name $instruction_name \
    --compressor_hidden_size $compressor_hidden_size \
    --lm_model_hidden_size $lm_model_hidden_size \
    --num_compressor_layers $num_compressor_layers \
    --num_compressor_encoder_layers $num_compressor_encoder_layers \
    --random_num_documents \
    --max_num_documents $max_num_documents \
    --min_num_documents $min_num_documents \
    --pool_window_size $pool_window_size \
    --train_batch_size $batch_size \
    --eval_batch_size $batch_size \
    --max_steps $max_steps \
    --dev_steps $dev_steps \
    --test_steps $test_steps \
    --save_steps $save_steps \
    --logging_steps $logging_steps \
    --do_benchmark \
    --benchmark_dev_steps $benchmark_dev_steps \
    --benchmark_test_steps $benchmark_test_steps \
    --benchmark_metric $benchmark_metric \
    --gold_first_for_kd \
    --random_pool_window_size \
    --cand_pool_window_sizes ${cand_pool_window_sizes[@]} \
    | tee ${output_dir}/train.log