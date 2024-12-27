import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from dataclasses import dataclass
from typing import Optional
# Class to represent the model parameters

@dataclass
class ModelArgs:
    dim: int = 4096
    n_layers: int = 32
    n_heads: int = 32                 # For the query
    n_kv_heads: Optional[int] = None  # For the keys and values
    vocab_size: int = -1              # Will be set when we load the tokenizer
    multiple_of: int = 256            # These two will be used in feed forward network
    ffn_dim_multiplyer: Optional[float] = None
    norm_eps: float = 1e-5
    max_batch_size: int = 32
    max_seq_len: int = 2048
    device: str = None
# torch.arange(0, head_dim, 2).float()
# The above code will generate numbers as [0, 2, 4, 6, 8, ..., head_dim - 2]
# That list wont contain head dim

# Consider the formuala
# 2(i-1) for i = [1, 2, 3, 4, 5, ..., dim/2]
# The above formula would generate numbers as [((1-1)*2), ((2-1)*2), ((3-1)*2), ..., ((dim/2 - 1)*2)]
# Which is equivalent to [0, 2, 4, ..., dim-2]
# This can be achieved from the code torch.arange(0, dim, 2).float()
def precompute_theta_pos_frequencies(head_dim: int, seq_len: int, device: str, theta: float = 10000.0):

    assert head_dim % 2 == 0, "We can apply rotary position embeddings only to even dim token embeddings"

    #Now we build the theta parameters
    # According to the formula theta_i = 10000 ^ (-2(i-1)/dim), i = [1, 2, ..., dim/2] 
    # shape: (head_dim / 2)
    theta_numerator = torch.arange(0, head_dim, 2).float()
    # shape -> (head_dim/2)
    # The two in the above line is given by 2 the code for theta_numerator
    theta = 1.0 / (theta ** (theta_numerator / head_dim)).to(device)

    #Construct the positions ( the "m" parameter of the formula to convert the token embeddings into some form)
    # shape -> (Seq_Len)
    m = torch.arange(seq_len, device=device)

    # Multiply each theta by each position using the outer product
    # shape -> (Seq_Len) outer_product (Head_dim/2) -> (Seq_Len, Head_dim/2)

    freqs = torch.outer(m, theta)
    
    # We can compute complex numbers in the polar form = R * exp(i * m * theta), where R = 1 as follows
    # Shape -> (Seq_Len, head_dim/2) -> (Seq_Len, head_dim/2)
    freqs_complex = torch.polar(torch.ones_like(freqs), freqs)

    return freqs_complex
# The freqs_complex mention in function head is only the corresponding row to the position of x in the sequence or sentence.

def apply_rotary_embeddings(x: torch.Tensor, freqs_complex: torch.Tensor, device: str):
    # shape (B, Seq_Len, H, Head_dim) -> (B, Seq_Len, H, Head_dim/2)
    x_complex = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
    # shape (Seq_Len, Head_dim/2) -> (1, Seq_Len, 1, Head_dim / 2)
    freqs_complex = freqs_complex.unsqueeze(0).unsqueeze(2)
    # shape (B, Seq_Len, H, Head_Dim / 2) * (1, Seq_Len, 1, Head_Dim / 2) = (B, Seq_Len, H, Head_dim/2)
    x_rotated = x_complex * freqs_complex
    # shape (B, Seq_Len, H, Head_dim/2) -> (B, Seq_Len, H, Head_Dim/2, 2)
    x_out = torch.view_as_real(x_rotated)
    # shape (B, Seq_Len, H, Head_Dim/2, 2) -> (B, Seq_Len, H, Head_Dim)
    x_out = x_out.reshape(*x.shape)

    return x_out.type_as(x).to(device)
class RMSNorm(nn.Module):

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()

        self.eps = eps
        # Then we introduce a gamma parameter
        self.weight = nn.Parameter(torch.ones(dim))
    
    def _norm(self, x: torch.Tensor):
        # shape (B, Seq_Len, Dim) * (B, Seq_Len, 1) = (B, Seq_Len, Dim)
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
    
    def forward(self, x: torch.Tensor):
        # shape (Dim) * (B, Seq_Len, Dim) = (B, Seq_Len, Dim)
        return self.weight* self._norm(x.float()).type_as(x)
def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    batch_size, seq_len, n_kv_heads, n_head_dim = x.shape
    if n_rep == 1:
        return x
    else:
        # shape (B, Seq_Len, N_KV_Heads, Head_dim)
        (x[:, :, :, None, :].expand(batch_size, seq_len, n_kv_heads, n_rep, n_head_dim).reshape(batch_size, seq_len, n_kv_heads*n_rep, n_head_dim))
# This code will work only for inferencing

class SelfAttention(nn.Module):

    def __init__(self, args: ModelArgs):
        super().__init__()

        self.n_kv_heads = args.n_heads if args.n_kv_heads is None else args.n_kv_heads
        self.n_heads_q = args.n_heads
        self.n_rep = self.n_heads_q // self.n_kv_heads
        
        self.head_dim = args.dim // args.n_heads

        self.wq = nn.Linear(args.dim, self.n_heads*self.head_dim, bias=False)
        self.wk = nn.Linear(args.dim, self.n_kv_heads*self.head_dim, bias=False)
        self.wv = nn.Linear(args.dim, self.n_kv_heads*self.head_dim, bias=False)
        self.wo = nn.Linear(self.n_heads*self.head_dim, args.dim, bias=False)

        self.cache_k = torch.zeros((args.max_batch_size, args.max_seq_len, self.n_kv_heads, self.head_dim))
    
    def forward(self, x: torch.Tensor, start_pos: int, freqs_complex: torch.Tensor):
        # (B, Seq_Len, Dim)
        batch_size, seq_len = x.shape
        # shape (B, Seq_Len, Dim) -> (B, Seq_Len, H_q * Head_Dim)
        xq = self.wq(x)
        # shape (B, Seq_Len, Dim) -> (B, Seq_Len, H_KV * Head_dim)    H_KV may be smaller than H_Q
        xk = self.wk(x)
        xv = self.wv(x)

        # shape (B, Seq_Len, H_q * Head_dim) -> (B, Seq_LEn, H_q, Head_dim)
        xq = xq.view(batch_size, seq_len, self.n_heads_q, self.head_dim)
        xk = xk.view(batch_size, seq_len, self.n_kv_heads, self.head_dim)
        # shape (B, Seq_Len, H_kv * head_dim) -> (B, Seq_Len, H_kv, head_dim)
        xv = xv.view(batch_size, seq_len, self.n_kv_heads, self.head_dim)

        # Now we have to apply rotary positional embeddings to only query and keys

        xq = apply_rotary_embeddings(xq, freqs_complex, device=x.device)
        xk = apply_rotary_embeddings(xk, freqs_complex, device=x.device)

        # Replace the entry in the cache for this token
        self.cache_k[:batch_size, start_pos: start_pos+seq_len] = xk
        self.cache_v[:batch_size, start_pos: start_pos+seq_len] = xv

        # Retrieve all the cached keys and values so far
        # shape (B, Seq_Len_KV, H_Kv, Head_dim)
        keys = self.cache_k[:batch_size, 0:start_pos+seq_len]
        values = self.cache_v[:batch_size, 0:start_pos+seq_len]

        # Repeat the heads of the k, v to match the number of heads of query
        keys = repeat_kv(keys, self.n_rep)
        values = repeat_kv(values, self.n_rep)

        # shape (B, Seq_Len, H_q, Head_dim) ->  (B, H_q, seq_len, Head_dim)    Here seq_len will be 1
        xq = xq.transpose(1, 2)
        keys = keys.transpose(1, 2)
        values = values.transpose(1, 2)

        # shape (B, H_q, 1, head_dim) @ (B, H_q, head_dim, Seq_Len_KV) = (B, H_Q, 1, Seq_Len_KV)
        scores = torch.matmul(xq, keys.transpose(2, 3) / math.sqrt(self.head_dim))
        scores = F.softmax(scores.float(), dim=-1).type_as(xq)

        # shape (B, H_q, 1, Head_dim) @ (B, H_q, Seq_Len_KV, head_dim)
        output = torch.matmul(scores, values)

        # shape (B, H_q, 1, head_dim) -> (B, 1, H_q, head_dim) -> (B, 1, dim)
        output = (output.transpose(1,2).contiguous().view(batch_size, seq_len, -1))

        # shape (B, 1, dim) -> (B, 1, dim)
        return self.wo(output)



class FeedForward(nn.Module):

    def __init__(self, args: ModelArgs):
        super().__init__()

        hidden_dim = 4 * args.dim
        hidden_dim = int((2 * hidden_dim / 3))
        
        if args.ffn_dim_multiplyer is not None:
            hidden_dim = int(args.ffn_dim_multiplyer * hidden_dim)
        
        # Now we have to round the hidden dim to nearest multiple of the multiplier parameter
        hidden_dim = args.multiple_of * ((hidden_dim + args.multiple_of - 1) // args.multiple_of)

        self.w1 = nn.Linear(args.dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, args.dim, bias=False)
        self.w3 = nn.Linear(args.dim, hidden_dim, bias=False)
    
    def forward(self, x:torch.Tensor):
        swish = F.silu(self.w1(x))
        x_v = self.w3(x)
        x = swish * x_v
        x = self.w2(x)

        return x
class EncoderBlock(nn.Module):

    def __init__(self, args: ModelArgs):
        super().__init__()

        self.n_heads = args.n_heads
        self.dim = args.dim
        self.head_dim = args.dim // args.n_heads

        self.attention = SelfAttention()
        self.feed_forward = FeedForward()

        # Normalization before attention mechanism
        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
        # Normalization before feed forward network
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)
    
    def forward(self, x: torch.Tensor, start_pos: int, freqs_complex: torch.Tensor):
        # shape (B, Seq_Len, Dim) + (B, Seq_Len, Dim) -> (B, Seq_Len, Dim)
        h = x + self.attention.forward(self.attention_norm(x), start_pos, freqs_complex)
        out = h + self.feed_forward.forward(self.ffn_norm(h))
        return out
# The class below would represent the transformer block which consists of all the parts of llama model except the softmax block

class Transformer(nn.Module):

    def __init__(self, args: ModelArgs):
        super().__init__()

        assert args.vocab_size != -1, 'we confirm that the vocab size is set'

        self.args = args
        self.vocab_size = args.vocab_size
        self.n_layers = args.n_layers
        self.tok_embeddings = nn.Embedding(self.vocab_size, args.dim)
        
        self.layers = nn.ModuleList()
        for _ in range(self.n_layers):
            self.layers.append(EncoderBlock(args))
        
        self.norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.output = nn.Linear(args.dim, self.vocab_size, bias=False)

        self.freqs_complex = precompute_theta_pos_frequencies(self.args.dim // self.args.n_heads, self.args.max_seq_len*2, device=self.args.device)

    def forward(self, tokens: torch.Tensor, start_pos: int):
        # (B, Seq_Len)
        batch_size, seq_len = tokens.shape
    
        assert seq_len == 1, "Only one token is passes to the model as we implement the KV cache"

        # (B, Seq_Len) -> (B, Seq_Len, dim)
        h = self.tok_embeddings(tokens)

        # Here we precompute some information related to positions that we then give to successive layers
        #Retrieve the pairs (m, theta) corresponding to the positions [start_pos, start_pos]
        freqs_complex = self.freqs_complex[start_pos:start_pos+seq_len]

        # Consecutively apply all the encoder layers
        for layer in self.layers:
            h = layer(h, start_pos, freqs_complex)
        
        # Then we apply RMSNorm to the output that is obtained from all the encoder blocks
        h = self.norm(h)

        output = self.output(h).float()

        return output



