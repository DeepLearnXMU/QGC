from typing import Optional, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import LlamaForCausalLM, LlamaPreTrainedModel, LlamaModel
from src.generation_utils import CCGenerationMixin


def expand_mask(mask: torch.Tensor, dtype: torch.dtype, tgt_len: Optional[int] = None):
    bsz, src_len = mask.size()
    tgt_len = tgt_len if tgt_len is not None else src_len
    expanded_mask = mask[:, None, None, :].expand(bsz, 1, tgt_len, src_len).to(dtype)
    inverted_mask = 1.0 - expanded_mask
    return inverted_mask.masked_fill(inverted_mask.to(torch.bool), torch.finfo(dtype).min)


class ModelWithQGC(LlamaPreTrainedModel, CCGenerationMixin):
    def __init__(self, args, compressor: LlamaModel, pooling_layer, lm_model: LlamaForCausalLM):
        super().__init__(lm_model.config)
        self.args = args
        
        self.compressor = compressor
        self.lm_model = lm_model
        self.pooling_layer = pooling_layer
        self.semantic_alignment_layer = nn.Linear(args.compressor_hidden_size, args.lm_model_hidden_size)

        for param in self.lm_model.parameters():
            param.requires_grad = False

        if args.fix_compressor_mlp_parameters:
            for name, param in self.compressor.named_parameters():
                if 'mlp' in name:
                    param.requires_grad = False

        
    @property
    def llm_embed_tokens(self):
        return self.lm_model.get_input_embeddings()

    def context_encoder(self, input_ids, attention_mask):
        position_ids = attention_mask.long().cumsum(-1) - 1
        position_ids.masked_fill_(~attention_mask.bool(), 1)
        hidden_states = self.compressor.embed_tokens(input_ids)

        attention_mask = expand_mask(
            attention_mask, hidden_states.dtype, attention_mask.size(1)
        ).to(hidden_states.device)
        
        for idx in range(self.args.num_compressor_encoder_layers):
            layer = self.compressor.layers[idx]
            layer_outputs = layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
            )
            hidden_states = layer_outputs[0]
        return hidden_states
    
    def reviewing_layer(self, hidden_states, attention_mask):
        position_ids = attention_mask.long().cumsum(-1) - 1
        position_ids.masked_fill_(~attention_mask.bool(), 1)
        
        attention_mask = expand_mask(
            attention_mask, hidden_states.dtype, attention_mask.size(1)
        ).to(hidden_states.device)
        
        for idx in range(self.args.num_compressor_encoder_layers, self.args.num_compressor_layers):
            layer = self.compressor.layers[idx]
            layer_outputs = layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
            )
            hidden_states = layer_outputs[0]
        hidden_states = self.compressor.norm(hidden_states)
        return hidden_states

    def compress_doc(
        self, 
        enc_doc_tokens, 
        enc_doc_mask, 
        enc_que_tokens, 
        enc_que_mask,
        llm_pfx_tokens=None,
        llm_pfx_mask=None,
        **kwargs,
    ):
        doc_bsz = enc_doc_tokens.size(0)
        que_bsz = enc_que_tokens.size(0)

        if doc_bsz > que_bsz:
            repeat_n = doc_bsz // que_bsz
            enc_repeated_que_tokens = (enc_que_tokens[None, ...].repeat(repeat_n, 1, 1).view(-1, enc_que_tokens.size(1)))
            enc_repeated_que_mask = (enc_que_mask[None, ...].repeat(repeat_n, 1, 1).view(-1, enc_que_mask.size(1)))
            context_input_ids = torch.cat([enc_repeated_que_tokens, enc_doc_tokens], dim=1)
            context_attention_mask = torch.cat([enc_repeated_que_mask, enc_doc_mask], dim=1)
        else:
            enc_repeated_que_mask = enc_que_mask
            context_input_ids = torch.cat([enc_que_tokens, enc_doc_tokens], dim=1)
            context_attention_mask = torch.cat([enc_que_mask, enc_doc_mask], dim=1)

        context_hidden_states = self.context_encoder(context_input_ids, context_attention_mask)
        context_que_hidden_states = context_hidden_states[:, :enc_que_tokens.size(1)]
        context_doc_hidden_states = context_hidden_states[:, enc_que_tokens.size(1):]
        
        weighted_hidden_states, weighted_attention_mask = self.pooling_layer(
            que_hidden_states=context_que_hidden_states,
            doc_hidden_states=context_doc_hidden_states,
            enc_que_mask=enc_repeated_que_mask,
            enc_doc_mask=enc_doc_mask,
            **kwargs,
        )
        
        reviewing_input_hidden_states = torch.cat([context_hidden_states, weighted_hidden_states], dim=1)
        reviewing_input_attention_mask = torch.cat([context_attention_mask, weighted_attention_mask], dim=1)
        
        reviewing_hidden_states = self.reviewing_layer(reviewing_input_hidden_states, reviewing_input_attention_mask)
        final_hidden_states = self.semantic_alignment_layer(reviewing_hidden_states[:, -weighted_hidden_states.size(1):])
        
        prefix_embeds = self.llm_embed_tokens(llm_pfx_tokens)
        cmp_llm_hidden_states = torch.cat([prefix_embeds, final_hidden_states], dim=1)
        cmp_llm_attention_mask = torch.cat([llm_pfx_mask, weighted_attention_mask], dim=1)

        if doc_bsz > que_bsz:
            cmp_llm_hidden_states = cmp_llm_hidden_states.view(que_bsz, -1, cmp_llm_hidden_states.size(-1))
            cmp_llm_attention_mask = cmp_llm_attention_mask.view(que_bsz, -1)

        return cmp_llm_hidden_states, cmp_llm_attention_mask


    def construct_llm_inputs(
        self,
        llm_ins_tokens, 
        llm_ins_mask, 
        cmp_llm_doc_embeds, 
        cmp_llm_doc_mask,
        llm_que_tokens, 
        llm_que_mask, 
        llm_tgt_tokens, 
        llm_tgt_mask,
        **kwargs,
    ):
        llm_inputs_embeds = torch.cat(
            [
                self.llm_embed_tokens(llm_ins_tokens),
                cmp_llm_doc_embeds,
                self.llm_embed_tokens(llm_que_tokens),
                self.llm_embed_tokens(llm_tgt_tokens),
            ],
            dim=1,
        )
        llm_attention_mask = torch.cat([llm_ins_mask, cmp_llm_doc_mask, llm_que_mask, llm_tgt_mask], dim=1)
        llm_position_ids = llm_attention_mask.long().cumsum(-1) - 1
        llm_position_ids.masked_fill_(~llm_attention_mask.bool(), 1)

        llm_labels = torch.full_like(llm_attention_mask, self.args.label_pad_token_id)
        llm_labels[:, -llm_tgt_tokens.size(1):] = llm_tgt_tokens.masked_fill(
            ~llm_tgt_mask.bool(), self.args.label_pad_token_id,
        )

        return {
            'inputs_embeds': llm_inputs_embeds,
            'attention_mask': llm_attention_mask,
            'position_ids': llm_position_ids,
            'labels': llm_labels,
        }


    def construct_llm_inputs_for_generation(
        self,
        llm_ins_tokens, 
        llm_ins_mask, 
        cmp_llm_doc_embeds, 
        cmp_llm_doc_mask,
        **kwargs,
    ):
        llm_inputs_embeds = torch.cat([self.llm_embed_tokens(llm_ins_tokens), cmp_llm_doc_embeds], dim=1)
        llm_attention_mask = torch.cat([llm_ins_mask, cmp_llm_doc_mask], dim=1)
        llm_position_ids = llm_attention_mask.long().cumsum(-1) - 1
        llm_position_ids.masked_fill_(llm_attention_mask == 0, 1)

        return {
            'inputs_embeds': llm_inputs_embeds,
            'attention_mask': llm_attention_mask,
            'position_ids': llm_position_ids,
        }


    @torch.no_grad()
    def get_text_logits(
        self,
        llm_ins_tokens, llm_ins_mask, llm_doc_tokens, llm_doc_mask,
        llm_que_tokens, llm_que_mask, llm_tgt_tokens, llm_tgt_mask,
    ):
        text_llm_input_ids = torch.cat([llm_ins_tokens, llm_doc_tokens, llm_que_tokens, llm_tgt_tokens], dim=1)
        text_llm_attention_mask = torch.cat([llm_ins_mask, llm_doc_mask, llm_que_mask, llm_tgt_mask], dim=1)
        text_llm_position_ids = text_llm_attention_mask.long().cumsum(-1) - 1
        text_llm_position_ids.masked_fill_(text_llm_attention_mask == 0, 1)

        text_llm_outputs = self.lm_model(
            input_ids=text_llm_input_ids,
            attention_mask=text_llm_attention_mask,
            position_ids=text_llm_position_ids,
        )
        text_logits = text_llm_outputs.logits[:, -llm_tgt_tokens.size(1):]
        return text_logits

    
    def joint_forward(
        self,
        enc_doc_tokens, 
        enc_doc_mask, 
        enc_que_tokens, 
        enc_que_mask,
        llm_ins_tokens, 
        llm_ins_mask, 
        llm_doc_tokens, 
        llm_doc_mask,
        llm_que_tokens, 
        llm_que_mask, 
        llm_tgt_tokens, 
        llm_tgt_mask,
        llm_pfx_tokens=None,
        llm_pfx_mask=None,
        **kwargs,
    ):
        cmp_llm_doc_embeds, cmp_llm_doc_mask = self.compress_doc(
            enc_doc_tokens, enc_doc_mask, enc_que_tokens, enc_que_mask,
            llm_pfx_tokens=llm_pfx_tokens, llm_pfx_mask=llm_pfx_mask,
        )
        
        text_logits = self.get_text_logits(
            llm_ins_tokens, llm_ins_mask, llm_doc_tokens, llm_doc_mask,
            llm_que_tokens, llm_que_mask, llm_tgt_tokens, llm_tgt_mask,
        )
        
        embed_llm_inputs = self.construct_llm_inputs(
            llm_ins_tokens, llm_ins_mask, cmp_llm_doc_embeds, cmp_llm_doc_mask,
            llm_que_tokens, llm_que_mask, llm_tgt_tokens, llm_tgt_mask,
        )
        embed_llm_output = self.lm_model(**embed_llm_inputs)
        embed_logits = embed_llm_output.logits[:, -llm_tgt_tokens.size(1):]

        distillation_loss = F.kl_div(
            F.log_softmax(embed_logits, dim=-1),
            F.softmax(text_logits, dim=-1),
            reduction='none',
        )
        distillation_loss = distillation_loss.sum(dim=-1).masked_fill(~llm_tgt_mask.bool(), 0.0)
        distillation_loss = distillation_loss.sum() / llm_tgt_mask.sum()
    
        return embed_llm_output.loss, distillation_loss
    

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        position_ids=None,
        first_time=False,
        **kwargs,
    ):
        if past_key_values and not first_time:
            input_ids = input_ids[:, -1:]
        
        if attention_mask is not None and position_ids is None:
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)

            if past_key_values:
                position_ids = position_ids[:, -input_ids.size(1):]

        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
                "position_ids": position_ids,
            }
        )
        return model_inputs
    
    
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        return self.lm_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )