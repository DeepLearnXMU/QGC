batch_size=4
lm_model_name=longchat
compressor_hidden_size=4096
lm_model_hidden_size=5120
data_path=/path-to-data
compressor_path=/path-to-compressor
lm_model_path=/path-to-longchat-13b
num_compressor_layers=4
num_compressor_encoder_layers=2
from_checkpoint=/path-to-checkpoint
benchmark_metric=accuracy
num_eval_documents=4
pw_window_sizes=(2 4 6 8)
pw_window_sizes_str=$(printf "_%s" "${pw_window_sizes[@]}")
pw_window_sizes_str=${pw_window_sizes_str:1}

echo num_eval_documents=${num_eval_documents}-pw=${pw_window_sizes_str}
save_path=/path-to-save-generation
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
    | tee ${save_path}/inference.log
