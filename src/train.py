import os
import jsonlines
from tqdm.auto import tqdm

import torch
import transformers
from accelerate import Accelerator

from src.args import JointArguments
from src.model import ModelWithQGC
from src.dataset import TrainDataset
from src.utils.constant import *
from src.utils.logger import get_logger
from src.utils.metrics import benchmark_function_map
from pooling_layers import PoolingLayer

logger = get_logger(__name__)
def get_model_param_count(model, trainable_only):
    param_count = 0
    for param in model.parameters():
        if not trainable_only or param.requires_grad:
            param_count += param.numel()
    return param_count

def load_dataloader(args: JointArguments, enc_tokenizer, llm_tokenizer, split):
    if hasattr(args, f'{split}_data_path') and getattr(args, f'{split}_data_path') != None:
        filepath = getattr(args, f'{split}_data_path')
    else:
        filepath = os.path.join(args.data_path, f'{split}.jsonl')
    if not os.path.isfile(filepath):
        return None

    is_training = split == 'train'
    logger.info(f'load {split} data from {filepath}')

    dataset = TrainDataset(
        filepath=filepath,
        enc_tokenizer=enc_tokenizer,
        llm_tokenizer=llm_tokenizer,
        max_doc_tokens=args.max_doc_tokens,
        que_mask_ratio=args.question_mask_ratio if is_training else None,
        max_num_documents=args.max_num_documents,
        min_num_documents=args.min_num_documents,
        random_num_documents=args.random_num_documents,
        num_gold_documents=args.num_gold_documents,
        use_answer_as_target=args.use_answer_as_target,
        instruction_name=args.instruction_name,
        gold_first_for_kd=args.gold_first_for_kd,
    )
    
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.train_batch_size if is_training else args.eval_batch_size,
        shuffle=is_training,
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

    logger.info('load dataset ...')
    train_dataloader = load_dataloader(args, enc_tokenizer, llm_tokenizer, 'train')
    dev_dataloader = load_dataloader(args, enc_tokenizer, llm_tokenizer, 'dev')
    test_dataloader = load_dataloader(args, enc_tokenizer, llm_tokenizer, 'test')

    if args.generation_split_token is None:
        args.generation_split_token = (
            '</s>' if args.lm_model_name == 'longchat' or 'natural_questions' in args.data_path
            else '\n\n'
        )

    accelerator = Accelerator()
    device = accelerator.device
    
    logger.info('load compressor ...')
    compressor_config = transformers.AutoConfig.from_pretrained(args.compressor_path)
    compressor_config.num_hidden_layers = args.num_compressor_layers
    compressor = transformers.LlamaModel.from_pretrained(args.compressor_path, config=compressor_config)
    compressor.resize_token_embeddings(len(enc_tokenizer))
    with torch.no_grad():
        compressor.get_input_embeddings().weight[-len(additional_special_tokens):] \
            = compressor.get_input_embeddings().weight[:-len(additional_special_tokens)].mean(dim=0)

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

    pooling_layer = PoolingLayer(args)
    model = ModelWithQGC(args, compressor=compressor, pooling_layer=pooling_layer, lm_model=lm_model)
    
    max_steps = args.max_steps // accelerator.num_processes
    num_examples = len(train_dataloader)
    total_train_batch_size = args.train_batch_size * args.gradient_accumulation_steps
    num_update_steps_per_epoch = num_examples // args.gradient_accumulation_steps
    num_update_steps_per_epoch = max(num_update_steps_per_epoch, 1)
    num_train_epochs = max_steps // num_update_steps_per_epoch + int(
        max_steps % num_update_steps_per_epoch > 0
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    lr_scheduler = transformers.get_scheduler(
        args.lr_scheduler_type, optimizer=optimizer,
        num_warmup_steps=args.get_warmup_steps(max_steps), num_training_steps=max_steps,
    )

    if args.from_checkpoint is not None:
        logger.info(f'load model checkpoint from {args.from_checkpoint}')
        model.semantic_alignment_layer.load_state_dict(torch.load(os.path.join(args.from_checkpoint, FFN_WEIGHTS_NAME), map_location='cpu'))
        model.compressor.load_state_dict(torch.load(os.path.join(args.from_checkpoint, COMPRESSOR_WEIGHTS_NAME), map_location='cpu'))
        model.pooling_layer.load_state_dict(torch.load(os.path.join(args.from_checkpoint, POOLING_WEIGHTS_NAME), map_location='cpu'))


    model, optimizer, train_dataloader, dev_dataloader, test_dataloader = \
        accelerator.prepare(model, optimizer, train_dataloader, dev_dataloader, test_dataloader)

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {num_examples:,}")
    logger.info(f"  Num Epochs = {num_train_epochs:,}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size:,}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_train_batch_size:,}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {max_steps:,}")
    logger.info(f"  Number of trainable parameters = {get_model_param_count(model, trainable_only=True):,}")
    logger.info(f"  Model Structure = {model}")

    global_step = 0
    current_step = 0
    global_epoch = 0
    global_step_last_logged = 0
    tr_loss = torch.tensor(0.0).to(device)
    tr_ce_loss = torch.tensor(0.0).to(device)
    tr_kd_loss = torch.tensor(0.0).to(device)

    def training_step(model, inputs):
        model.train()
        ce_loss, kd_loss = model.joint_forward(**inputs)
        loss = ce_loss + kd_loss
        accelerator.backward(loss)

        return [
            loss_item / args.gradient_accumulation_steps
            for loss_item in [loss, ce_loss, kd_loss]
        ]

    @torch.no_grad()
    def prediction_step(model, inputs):
        model.eval()
        ce_loss, kd_loss = model.joint_forward(**inputs)
        loss = ce_loss + kd_loss
        return loss, ce_loss, kd_loss

    @torch.no_grad()
    def benchmark_step(model, inputs):        
        model.eval()
        benchmark_answers = inputs['benchmark']
        
        cmp_llm_doc_embeds, cmp_llm_doc_mask = model.compress_doc(**inputs, window_size=args.pool_window_size)
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

        raw_generations = [
            element.split('Answer:')[1] for element in llm_tokenizer.batch_decode(outputs)
        ]

        # longchat use </s> to represent end, but llama use \n\n
        generations = [elem.strip().split(args.generation_split_token)[0] for elem in raw_generations]
        benchmark_function = benchmark_function_map[args.benchmark_metric]
        score_values = [benchmark_function(generation, answer) for generation, answer in zip(generations, benchmark_answers)]
        scores = torch.tensor(score_values, device=device)

        benchmark_outputs = [
            {
                'document': document,
                'raw_generation': raw_generation,
                'ext_generation': ext_generation,
                'answers': answers,
                'score': score
            }
            for document, raw_generation, ext_generation, answers, score in zip(
                llm_tokenizer.batch_decode(inputs['llm_doc_tokens']),
                llm_tokenizer.batch_decode(outputs),
                generations,
                benchmark_answers,
                score_values,
            )
        ]
        return scores, benchmark_outputs
    
    def evaluate(model, dataloader, prefix='eval'):
        evaluate_bar = tqdm(
            total=len(dataloader), leave=True, dynamic_ncols=True,
            disable=not accelerator.is_main_process, desc='evaluate'
        )
        model.eval()
        losses_host = ()
        ce_losses_host = ()
        kd_losses_host = ()
        for inputs in dataloader:
            bsz = inputs['llm_ans_tokens'].size(0)
            loss, ce_loss, kd_loss = prediction_step(model, inputs)
            losses_host += (accelerator.gather_for_metrics(loss.repeat(bsz)),)
            ce_losses_host += (accelerator.gather_for_metrics(ce_loss.repeat(bsz)),)
            kd_losses_host += (accelerator.gather_for_metrics(kd_loss.repeat(bsz)),)
            evaluate_bar.update(1)
        
        evaluate_bar.close()
        eval_loss = torch.cat(losses_host, dim=0)
        eval_ce_loss = torch.cat(ce_losses_host, dim=0)
        eval_kd_loss = torch.cat(kd_losses_host, dim=0)
        return {
            f'{prefix}_loss': round(eval_loss.mean().item(), 4),
            f'{prefix}_ce_loss': round(eval_ce_loss.mean().item(), 4),
            f'{prefix}_kd_loss': round(eval_kd_loss.mean().item(), 4),
        }
    
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

    def save_checkpoint(model, dev_benchmark_outputs=None, test_benchmark_outputs=None):
        checkpoint_folder = f'{PREFIX_CHECKPOINT_DIR}-{global_step}'
        output_dir = os.path.join(args.output_dir, checkpoint_folder)
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Saving model checkpoint to {output_dir}")
        torch.save(args, os.path.join(output_dir, TRAINING_ARGS_NAME))

        compressor_state_dict = accelerator.get_state_dict(model.compressor)
        pooling_layer_state_dict = accelerator.get_state_dict(model.pooling_layer)
        ffn_state_dict = accelerator.get_state_dict(model.semantic_alignment_layer)
        torch.save(compressor_state_dict, os.path.join(output_dir, COMPRESSOR_WEIGHTS_NAME))
        torch.save(pooling_layer_state_dict, os.path.join(output_dir, POOLING_WEIGHTS_NAME))
        torch.save(ffn_state_dict, os.path.join(output_dir, FFN_WEIGHTS_NAME))
    
        if accelerator.is_main_process:
            torch.save(optimizer.state_dict(), os.path.join(output_dir, OPTIMIZER_NAME))
            torch.save(lr_scheduler.state_dict(), os.path.join(output_dir, SCHEDULER_NAME))

        if dev_benchmark_outputs is not None:
            with jsonlines.open(os.path.join(output_dir, f'dev_benchmark.{accelerator.process_index}.jsonl'), 'w') as fw:
                for element in dev_benchmark_outputs:
                    fw.write(element)
        
        if test_benchmark_outputs is not None:
            with jsonlines.open(os.path.join(output_dir, f'test_benchmark.{accelerator.process_index}.jsonl'), 'w') as fw:
                for element in test_benchmark_outputs:
                    fw.write(element)

    model.train()
    model.zero_grad()
    total_batched_samples = 0
    trainning_bar = tqdm(total=max_steps, dynamic_ncols=True, disable=not accelerator.is_main_process, desc='train')
    for epoch in range(num_train_epochs):
        epoch_iterator = train_dataloader
        steps_in_epoch = len(epoch_iterator)

        step = -1
        for step, inputs in enumerate(epoch_iterator):
            total_batched_samples += 1
            with accelerator.accumulate(model):
                tr_loss_step, tr_ce_loss_step, tr_kd_loss_step = training_step(model, inputs)
            
            tr_loss += tr_loss_step
            tr_ce_loss += tr_ce_loss_step
            tr_kd_loss += tr_kd_loss_step

            if total_batched_samples % args.gradient_accumulation_steps == 0:
                if args.max_grad_norm is not None:
                    accelerator.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                
                optimizer.step()
                lr_scheduler.step()
                model.zero_grad()

                global_epoch = epoch + (step + 1) / steps_in_epoch
                global_step += 1
                trainning_bar.update(global_step - current_step)
                current_step = global_step

                logs = {
                    'step': global_step,
                    'epoch': round(global_epoch, 2),
                }
                if global_step % args.logging_steps == 0:
                    tr_loss_scalar = accelerator.gather(tr_loss).mean().item()
                    tr_ce_loss_scalar = accelerator.gather(tr_ce_loss).mean().item()
                    tr_kd_loss_scalar = accelerator.gather(tr_kd_loss).mean().item()

                    tr_loss -= tr_loss
                    tr_ce_loss -= tr_ce_loss
                    tr_kd_loss -= tr_kd_loss

                    logs.update(
                        {
                            'loss': round(tr_loss_scalar / (global_step - global_step_last_logged), 4),
                            'ce_loss': round(tr_ce_loss_scalar / (global_step - global_step_last_logged), 4),
                            'kd_loss': round(tr_kd_loss_scalar / (global_step - global_step_last_logged), 4),
                            'lr': round(lr_scheduler.get_last_lr()[0], 6),
                        }
                    )
                    global_step_last_logged = global_step
                    logger.info(logs)
                
                base_metrics = {
                    'step': global_step,
                    'epoch': round(global_epoch, 2),
                }
                if dev_dataloader is not None and global_step % args.dev_steps == 0:
                    dev_metrics = evaluate(model, dev_dataloader, prefix='dev')
                    dev_metrics.update(base_metrics)
                    logger.info(dev_metrics)
                
                if test_dataloader is not None and global_step % args.test_steps == 0:
                    test_metrics = evaluate(model, test_dataloader, prefix='test')
                    test_metrics.update(base_metrics)
                    logger.info(test_metrics)
                
                dev_benchmark_outputs = None
                if dev_dataloader is not None and args.do_benchmark and global_step % args.benchmark_dev_steps == 0:
                    dev_benchmark_metrics, dev_benchmark_outputs = benchmark(model, dev_dataloader, prefix='dev')
                    dev_benchmark_metrics.update(base_metrics)
                    logger.info(dev_benchmark_metrics)
                
                test_benchmark_outputs = None
                if test_dataloader is not None and args.do_benchmark and global_step % args.benchmark_test_steps == 0:
                    test_benchmark_metrics, test_benchmark_outputs = benchmark(model, test_dataloader, prefix='test')
                    test_benchmark_metrics.update(base_metrics)
                    logger.info(test_benchmark_metrics)

                if global_step % args.save_steps == 0:
                    save_checkpoint(model, dev_benchmark_outputs=dev_benchmark_outputs, test_benchmark_outputs=test_benchmark_outputs)
            
        if global_step > max_steps:
            trainning_bar.close()
            break

    logger.info("\n\nTraining completed. =)\n\n")
    save_checkpoint(model)


if __name__ == '__main__':
    parser = transformers.HfArgumentParser(JointArguments)
    args = parser.parse_args_into_dataclasses()[0]
    logger.info(args)
    main(args)