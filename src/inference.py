import os
import jsonlines
from tqdm.auto import tqdm

import torch
import transformers
from accelerate import Accelerator

from src.args import JointArguments
from src.model import ModelWithQGC
from src.dataset import InferDataset
from src.pooling_layers import InferPoolingLayer
from src.utils.constant import *
from src.utils.logger import get_logger
from src.utils.metrics import benchmark_function_map
logger = get_logger(__name__)


def load_dataloader(args: JointArguments, enc_tokenizer, llm_tokenizer):
    dataset = InferDataset(
        filepath=args.data_path,
        enc_tokenizer=enc_tokenizer,
        llm_tokenizer=llm_tokenizer,
        max_doc_tokens=args.max_doc_tokens,
        max_num_documents=args.num_eval_documents,
        llm_with_neg_documents=True,
        instruction_name=args.instruction_name,
    )
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.eval_batch_size,
        shuffle=False,
        collate_fn=dataset.collate_fn,
    )
    return dataloader

def main(args: JointArguments):
    transformers.trainer_utils.set_seed(args.seed)

    logger.info('load tokenizer ...')
    enc_tokenizer = transformers.AutoTokenizer.from_pretrained(args.compressor_path)
    llm_tokenizer = transformers.AutoTokenizer.from_pretrained(args.lm_model_path)
    enc_tokenizer.pad_token = enc_tokenizer.unk_token

    additional_special_tokens = ['<DOC>', '<QUE>', '<CMP>']
    enc_tokenizer.add_special_tokens({"additional_special_tokens": additional_special_tokens})
    
    generation_end_token = (
        '</s>' if args.lm_model_name == 'longchat' or 'natural_questions' in args.data_path
        else '\n\n'
    )

    logger.info('load dataset ...')
    test_dataloader = load_dataloader(args, enc_tokenizer, llm_tokenizer)

    accelerator = Accelerator()
    device = accelerator.device
    
    logger.info('load compressor ...')
    compressor_config = transformers.AutoConfig.from_pretrained(args.compressor_path)
    compressor_config.num_hidden_layers = args.num_compressor_layers
    compressor = transformers.LlamaModel.from_pretrained(args.compressor_path, config=compressor_config)
    compressor.resize_token_embeddings(len(enc_tokenizer))

    pooling_layer = InferPoolingLayer(args)

    logger.info('load lm_model ...')
    if args.lm_model_name == 'longchat':
        llm_config = transformers.AutoConfig.from_pretrained(args.lm_model_path)
        llm_config._flash_attn_2_enabled = True
        llm_config.use_cache = False
        from src.utils.llama_utils import replace_llama_with_condense
        replace_llama_with_condense(8)
        lm_model = transformers.LlamaForCausalLM.from_pretrained(args.lm_model_path, config=llm_config)

    elif args.lm_model_name == 'llama':
        llm_config = transformers.AutoConfig.from_pretrained(args.lm_model_path)
        llm_config._flash_attn_2_enabled = True
        llm_config.use_cache = False
        lm_model = transformers.LlamaForCausalLM.from_pretrained(args.lm_model_path, config=llm_config)
    
    else:
        raise NotImplementedError(args.lm_model_name)

    logger.info(f'build model and load checkpoint from {args.from_checkpoint}')
    model = ModelWithQGC(args, compressor=compressor, pooling_layer=pooling_layer, lm_model=lm_model)
    model.semantic_alignment_layer.load_state_dict(torch.load(os.path.join(args.from_checkpoint, FFN_WEIGHTS_NAME), map_location='cpu'))
    model.compressor.load_state_dict(torch.load(os.path.join(args.from_checkpoint, COMPRESSOR_WEIGHTS_NAME), map_location='cpu'))
    model.pooling_layer.load_state_dict(torch.load(os.path.join(args.from_checkpoint, POOLING_WEIGHTS_NAME), map_location='cpu'))

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    model, optimizer, test_dataloader = accelerator.prepare(model, optimizer, test_dataloader)

    logger.info(f'Model Structure = {model}')

    @torch.no_grad()
    def benchmark_step(model, inputs):        
        model.eval()
        benchmark_answers = inputs['benchmark']

        cmp_llm_doc_embeds, cmp_llm_doc_mask = model.compress_doc(**inputs)
        first_llm_inputs = model.construct_llm_inputs_for_generation(
            **inputs,
            cmp_llm_doc_embeds=cmp_llm_doc_embeds,
            cmp_llm_doc_mask=cmp_llm_doc_mask,
        )
        first_llm_outputs = model(**first_llm_inputs, use_cache=True)

        second_llm_inputs = {
            'input_ids': inputs['llm_que_tokens'],
            'attention_mask': torch.cat([first_llm_inputs['attention_mask'], inputs['llm_que_mask']], dim=1),
            'past_key_values': first_llm_outputs.past_key_values,
        }
        outputs = model.generate(**second_llm_inputs, do_sample=False, max_new_tokens=args.max_new_tokens, use_cache=True)
        context_length = second_llm_inputs['input_ids'].size(1)

        llm_generations = [elem.strip().split(generation_end_token)[0] for elem in llm_tokenizer.batch_decode(outputs[:, context_length:])]
        benchmark_function = benchmark_function_map[args.benchmark_metric]
        score_values = [benchmark_function(generation, answer) for generation, answer in zip(llm_generations, benchmark_answers)]
        scores = torch.tensor(score_values, device=device)

        benchmark_outputs = [
            {
                'document': document,
                'raw_generation': raw_generation,
                'ext_generation': ext_generation,
                'answers': answers,
                'score': score,
            }
            for document, raw_generation, ext_generation, answers, score in zip(
                llm_tokenizer.batch_decode(inputs['llm_doc_tokens']),
                llm_tokenizer.batch_decode(outputs),
                llm_generations,
                benchmark_answers,
                score_values,
            )
        ]
        return scores, benchmark_outputs
    

    def benchmark(model, dataloader, prefix='benchmark'):
        benchmark_bar = tqdm(
            total=len(dataloader), leave=True, dynamic_ncols=True,
            disable=not accelerator.is_main_process, desc='benchmark'
        )
        model.eval()
        scores_host = ()
        outputs_host = []

        for inputs in dataloader:
            scores, outputs = benchmark_step(model, inputs)
            scores_host += (accelerator.gather_for_metrics(scores),)
            outputs_host += outputs
            benchmark_bar.update(1)
        
        benchmark_bar.close()
        mean_scores = torch.cat(scores_host, dim=0).mean()
        return [
            {
                f'{prefix}_score': round(mean_scores.item(), 4),
            },
            outputs_host,
        ]

    benchmark_metrics, benchmark_outputs = benchmark(model, test_dataloader, prefix='test')
    logger.info(benchmark_metrics)

    if accelerator.is_main_process:
        print(benchmark_metrics)

    with jsonlines.open(os.path.join(args.save_path, f'benchmark.{accelerator.process_index}.jsonl'), 'w') as fw:
        for element in benchmark_outputs:
            fw.write(element)

if __name__ == '__main__':
    parser = transformers.HfArgumentParser(JointArguments)
    args = parser.parse_args_into_dataclasses()[0]
    logger.info(args)
    main(args)