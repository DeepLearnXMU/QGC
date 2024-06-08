import random
import math
import torch
import torch.nn as nn
from transformers.models.llama.modeling_llama import LlamaRMSNorm


def fix_window_size_pooling(hidden_states, attention_mask, weights):
    bsz, pooled_length, window_size, hidden_size = hidden_states.size()
    scatter_matrix = torch.zeros_like(attention_mask)
    scatter_matrix[..., ::window_size] = 1
    scatter_index = scatter_matrix.cumsum(dim=-1) - 1

    hidden_states_after_weighting = (hidden_states * weights).view(bsz, -1, hidden_size)
    pooling_hidden_states = torch.zeros([bsz, pooled_length, hidden_size], device=hidden_states.device).to(hidden_states.dtype)
    pooling_hidden_states.scatter_add_(1, scatter_index[..., None].repeat(1, 1, hidden_size), hidden_states_after_weighting)

    pooling_attention_mask = torch.zeros([bsz, pooled_length], device=hidden_states.device).to(attention_mask.dtype)
    pooling_attention_mask.scatter_add_(1, scatter_index, attention_mask)
    pooling_attention_mask = pooling_attention_mask.greater(0).to(attention_mask.dtype)
    
    return pooling_hidden_states, pooling_attention_mask


class PoolingLayer(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.hidden_size = args.compressor_hidden_size
        self.num_heads = args.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.doc_layernorm = LlamaRMSNorm(args.compressor_hidden_size, eps=1e-05)
        self.que_layernorm = LlamaRMSNorm(args.compressor_hidden_size, eps=1e-05)


    def forward(self, que_hidden_states, doc_hidden_states, enc_doc_mask, enc_que_mask, window_size=None, **kwargs):
        if window_size is None:
            if self.args.random_pool_window_size:
                window_size = random.choice(self.args.cand_pool_window_sizes)
            else:
                window_size = self.args.pool_window_size
        
        bsz, d_len, hidden_size = doc_hidden_states.size()
        if d_len % window_size != 0:

            def padding(tensor, shape):
                return torch.cat([torch.zeros(shape, dtype=tensor.dtype, device=tensor.device), tensor], dim=1)

            padding_length = window_size - d_len % window_size
            doc_hidden_states = padding(doc_hidden_states, shape=(bsz, padding_length, hidden_size))
            enc_doc_mask = padding(enc_doc_mask, shape=(bsz, padding_length))
            d_len = enc_doc_mask.size(1)

        doc_hidden_states = self.doc_layernorm(doc_hidden_states)
        que_mean_hidden_states = que_hidden_states.masked_fill(~enc_que_mask[..., None].bool(), 0.0)
        que_mean_hidden_states = que_mean_hidden_states.sum(dim=1) / enc_que_mask[..., None].sum(dim=1)
        que_mean_hidden_states = self.que_layernorm(que_mean_hidden_states)
        
        query_states = self.q_proj(que_mean_hidden_states).view(bsz, self.num_heads, self.head_dim)
        key_states = self.k_proj(doc_hidden_states).view(bsz, -1, self.num_heads, self.head_dim).transpose(1, 2)
        value_states = doc_hidden_states.view(bsz, -1, self.num_heads, self.head_dim).transpose(1, 2)

        pooling_weights = torch.einsum('bnh,bndh->bnd', query_states, key_states) / math.sqrt(self.head_dim)
        pooling_weights.masked_fill_(~enc_doc_mask.unsqueeze(1).bool(), torch.finfo(query_states.dtype).min)
        pooling_weights = pooling_weights.view(bsz, self.num_heads, -1, window_size)
        pooling_weights = pooling_weights.softmax(dim=-1, dtype=torch.float32).to(query_states.dtype)

        combined_pooling_weights = pooling_weights.permute(0, 2, 3, 1)
        combined_pooling_weights = combined_pooling_weights[..., None].repeat(1, 1, 1, 1, self.head_dim).view(bsz, -1, window_size, self.hidden_size)
        combined_value_states = value_states.permute(0, 2, 1, 3).view(bsz, -1, window_size, self.hidden_size)

        pooling_hidden_states, pooling_attention_mask = fix_window_size_pooling(
            hidden_states=combined_value_states,
            attention_mask=enc_doc_mask,
            weights=combined_pooling_weights,
        )
        
        return pooling_hidden_states, pooling_attention_mask


class InferPoolingLayer(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.hidden_size = args.compressor_hidden_size
        self.num_heads = args.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.doc_layernorm = LlamaRMSNorm(args.compressor_hidden_size, eps=1e-05)
        self.que_layernorm = LlamaRMSNorm(args.compressor_hidden_size, eps=1e-05)


    def forward(self, que_hidden_states, doc_hidden_states, enc_doc_mask, enc_que_mask, **kwargs):
        def left_padding(tensor, shape, dim):
            return torch.cat([torch.zeros(shape, dtype=tensor.dtype, device=tensor.device), tensor], dim=dim)

        pw_window_sizes = self.args.pw_window_sizes
        bsz, d_len, hidden_size = doc_hidden_states.size()
        doc_hidden_states = self.doc_layernorm(doc_hidden_states)
        que_mean_hidden_states = que_hidden_states.masked_fill(~enc_que_mask[..., None].bool(), 0.0)
        que_mean_hidden_states = que_mean_hidden_states.sum(dim=1) / enc_que_mask[..., None].sum(dim=1)
        que_mean_hidden_states = self.que_layernorm(que_mean_hidden_states)

        query_states = self.q_proj(que_mean_hidden_states).view(bsz, self.num_heads, self.head_dim)
        key_states = self.k_proj(doc_hidden_states).view(bsz, -1, self.num_heads, self.head_dim).transpose(1, 2)
        value_states = doc_hidden_states.view(bsz, -1, self.num_heads, self.head_dim).transpose(1, 2)

        pooling_weights = torch.einsum('bnh,bndh->bnd', query_states, key_states) / math.sqrt(self.head_dim)
        pooling_weights.masked_fill_(~enc_doc_mask.unsqueeze(1).bool(), torch.finfo(query_states.dtype).min)
        
        pooling_max_len = math.ceil(d_len / min(pw_window_sizes))
        pooling_hidden_states = torch.zeros([bsz, pooling_max_len, hidden_size], device=query_states.device).to(query_states.dtype)
        pooling_attention_mask = torch.zeros([bsz, pooling_max_len], device=query_states.device).to(enc_doc_mask.dtype)
        
        for index in range(self.args.num_eval_documents):   
            current_batch_size = bsz // self.args.num_eval_documents
            current_weight = pooling_weights[index::self.args.num_eval_documents]
            current_attention_mask = enc_doc_mask[index::self.args.num_eval_documents]
            current_value_states = value_states[index::self.args.num_eval_documents].permute(0, 2, 1, 3).view(current_batch_size, -1, hidden_size)
            current_window_size = pw_window_sizes[index % len(pw_window_sizes)]
            
            if current_weight.size(-1) % current_window_size != 0:
                padding_length = current_window_size - current_weight.size(-1) % current_window_size
                current_weight = left_padding(current_weight, shape=(current_batch_size, self.num_heads, padding_length), dim=2)
                current_attention_mask = left_padding(current_attention_mask, shape=(current_batch_size, padding_length), dim=1)
                current_value_states = left_padding(current_value_states, shape=(current_batch_size, padding_length, hidden_size), dim=1)
            
            current_weight = current_weight.view(current_batch_size, self.num_heads, -1, current_window_size)        
            current_weight = current_weight.softmax(dim=-1, dtype=torch.float32).to(query_states.dtype)
            current_softmax_weight = current_weight.permute(0, 2, 3, 1)
            current_softmax_weight = current_softmax_weight[..., None].repeat(1, 1, 1, 1, self.head_dim).view(current_batch_size, -1, current_window_size, hidden_size)
            current_value_states = current_value_states.view(current_batch_size, -1, current_window_size, hidden_size)
            current_weight_hidden_states = (current_softmax_weight * current_value_states).sum(dim=2)
            current_attention_mask = current_attention_mask.view(current_batch_size, -1, current_window_size).sum(dim=-1).greater(0).to(enc_doc_mask.dtype)
            
            current_len = current_weight_hidden_states.size(1)
            pooling_hidden_states[index::self.args.num_eval_documents, -current_len:] = current_weight_hidden_states
            pooling_attention_mask[index::self.args.num_eval_documents, -current_len:] = current_attention_mask

        return pooling_hidden_states, pooling_attention_mask