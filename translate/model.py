from dataclasses import dataclass
import torch
from torch import nn
import torch.nn.functional as F

class LayerNorm(nn.Module):
    """ LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False """

    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)

class MLP(nn.Module):
    """ Simple Feed Forward Network with GELU activation. """

    def __init__(self, config):
        super().__init__()
        # contextual fully connected.
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        # activation function.
        self.gelu    = nn.GELU()
        # contextual projection.
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x

@dataclass
class T5Config:
    block_size: int = 1024
    vocab_size: int = 50304 # GPT-2 vocab_size of 50257, padded up to nearest multiple of 64 for efficiency
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = True # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster

class T5(nn.Module):
    def __init__(self, config: T5Config):
        self.config = config

    def forward(self, input_ids, attention_mask):
        pass

    def save_pretrained(self, save_directory):
        pass

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path):
        pass

    def generate(self, input_ids, attention_mask):
        pass
