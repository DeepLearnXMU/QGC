data_path=/path-to-test-data-file
compressor_path=/path-to-llama-2-7B
lm_model_path=/path-to-longchat-13B
from_checkpoint=/path-to-compressor-checkpoint
save_path=/path-to-save-generation

batch_size=4
lm_model_name=longchat
compressor_hidden_size=4096
lm_model_hidden_size=5120
num_compressor_layers=4
num_compressor_encoder_layers=2
benchmark_metric=accuracy
instruction_name=base
num_eval_documents=4
pw_window_sizes=(2 4 6 8)
pw_window_sizes_str=$(printf "_%s" "${pw_window_sizes[@]}")
pw_window_sizes_str=${pw_window_sizes_str:1}

mkdir -p $save_path

accelerate launch --config_file config/bf16.yaml \
    src/infer.py \
    --data_path $data_path \
    --compressor_path $compressor_path \
    --lm_model_name $lm_model_name \
    --lm_model_path $lm_model_path \
    --compressor_hidden_size $compressor_hidden_size \
    --lm_model_hidden_size $lm_model_hidden_size \
    --num_compressor_layers $num_compressor_layers \
    --num_compressor_encoder_layers $num_compressor_encoder_layers \
    --eval_batch_size $batch_size \
    --save_path $save_path \
    --num_eval_documents $num_eval_documents \
    --pw_window_sizes ${pw_window_sizes[@]} \
    --from_checkpoint $from_checkpoint \
    --benchmark_metric $benchmark_metric \
    --instruction_name $instruction_name \
    --fix_compressor_mlp_parameters \
    | tee ${save_path}/infer.log
