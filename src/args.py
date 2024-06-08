import math
from dataclasses import dataclass
from typing import List

@dataclass
class JointArguments:
    seed: int = 42

    # model args
    compressor_path: str = None
    lm_model_path: str = None
    lm_model_name: str = 'longchat'
    num_compressor_layers: int = 4
    num_compressor_encoder_layers: int = 2
    fix_compressor_mlp_parameters: bool = True
    num_attention_heads: int = 32
    attn_doc_topp: float = 0.25
    compressor_hidden_size: int = 4096
    lm_model_hidden_size: int = 5120
    
    # training args
    from_checkpoint: str = None
    generation_split_token: str = None

    pool_window_size: int = 4
    random_pool_window_size: bool = True
    cand_pool_window_sizes: List[int] = None

    train_batch_size: int = 4
    eval_batch_size: int = 4
    gradient_accumulation_steps: int = 4

    max_steps: int = 150000
    learning_rate: float = 5e-5
    lr_scheduler_type: str = 'linear'
    warmup_ratio: float = 0.0
    max_grad_norm: float = 1.0

    logging_steps: int = 100
    dev_steps: int = 500
    test_steps: int = 500
    save_steps: int = 1000

    do_benchmark: bool = True
    benchmark_dev_steps: int = 1000
    benchmark_test_steps: int = 1000
    benchmark_metric: str = None

    label_pad_token_id: int = -100

    # inference args
    pw_window_sizes: List[int] = None

    # data args
    save_path: str = None
    data_path: str = None
    output_dir: str = None
    train_data_path: str = None
    dev_data_path: str = None
    test_data_path: str = None
    num_eval_documents: int = 5

    num_gold_documents: int = 1
    use_answer_as_target: bool = False
    instruction_name: str = 'base'
    gold_first_for_kd: bool = True

    min_num_documents: int = 1
    max_num_documents: int = 5
    random_num_documents: bool = True

    max_new_tokens: int = 100
    max_doc_tokens: int = 512
    question_mask_ratio: float = 0.5

    def get_warmup_steps(self, num_training_steps):
        return math.ceil(num_training_steps * self.warmup_ratio)