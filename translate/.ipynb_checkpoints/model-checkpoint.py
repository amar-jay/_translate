from dataclasses import dataclass
import os
import math
import torch
from torch import nn
import torch.nn.functional as F
import json

@dataclass
class TConfig:
    block_size: int = 1024
    vocab_size: int = 50304 # GPT-2 vocab_size of 50257, padded up to nearest multiple of 64 for efficiency
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = True # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster
    padding_idx: int = 0 # padding token for input embeddings

    def save(self, save_directory):
        if not os.path.exists(save_directory):
            os.makedirs(save_directory)
        with open(os.path.join(save_directory, "config.json"), "w") as f:
            f.write(str(self))

    def load(self, save_directory):
        if not os.path.exists(save_directory):
            raise FileNotFoundError(f"Directory {save_directory} does not exist")
        with open(os.path.join(save_directory, "config.json"), "r") as f:
            data = json.load(f)
            self.__dict__.update(data)
        return self

    def __str__(self):
        return str(self.__dict__)

class LayerNorm(nn.Module):
    """ 
    LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False 

    Paper: https://arxiv.org/pdf/2002.04745
    It proposes that pre-LN stabizes the training process and leads to faster convergence.
    """


    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)

class MLP(nn.Module):
    """ Simple Feed Forward Network with GELU activation. """

    def __init__(self, config:TConfig):
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


class MultiheadAttention(nn.Module):
    """
    Multihead scaled dot-product attention with optional Flash Attention support.
    """
    def __init__(self, config):
        super().__init__()

        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch.
        # combined dense layer is more efficient and simple
        # silly reference: https://chatgpt.com/share/b6a00998-1151-45a9-ad96-310dcf7adeb6
        self.c_attn = nn.Linear(3 * config.n_embd, 3 * config.n_embd, bias=config.bias)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        # regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        self.block_size = config.block_size

        # flash attention make GPU go brrrrr but support is only in PyTorch >= 2.0
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        if not self.flash:
            print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")
            # causal mask to ensure that attention is only applied to the left in the input sequence

    def forward(self, queries, keys, values, mask=None):
        B, T, C = queries.size() # batch size, sequence length, embedding dimensionality (n_embd)
        assert T == keys.size(1) == values.size(1) == self.block_size
        assert self.n_embd == C

        # NOTE: I am contemplating whethere regsiter_buffer is necessary or not.
        if mask is None:
            self.register_buffer("bias", torch.tril(torch.ones(self.block_size, self.block_size))
                                        .view(1, 1, self.block_size, self.block_size))
        else:
            self.register_buffer("bias", mask)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v  = self.c_attn(torch.cat((queries, keys, values), dim=2)).split(self.n_embd, dim=2)


        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        if self.flash:
            # efficient attention using Flash Attention CUDA kernels
            y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=self.bias, dropout_p=self.dropout if self.training else 0)
        else:
            # manual implementation of attention
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
            att = F.softmax(att, dim=-1)
            y = self.attn_dropout(att)
            y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.c_proj(y))

        #NOTE: Research on how to get the attention weights from flash attention.
        return y

class EncoderBlock(nn.Module):
    def __init__(self, config: TConfig):
        super().__init__()
        # as said previously, pre-LN is better than post-LN.
        self.ln_1 = LayerNorm(config.n_embd, config.bias)
        self.attn = MultiheadAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, config.bias)
        self.mlp = MLP(config)
        self.drop = nn.Dropout(config.dropout)
    def forward(self, x, mask):
        # why an input mask is necessary?
        # in translation tasks, we need to mask out the padding tokens
        x = self.ln_1(x)
        x = x + self.drop(self.attn(x, x, x, mask))
        x = x + self.drop(self.mlp(self.ln_2(x)))
        return x

class Encoder(nn.Module):
    def __init__(self, config: TConfig):
        super().__init__()
        self.config = config
        self.encoder = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList([EncoderBlock(config) for _ in range(config.n_layer)]),
            ln_f = LayerNorm(config.n_embd, bias=config.bias),
        ))

    def forward(self, input_ids, input_mask):
        t = input_ids.size(1)
        device = input_ids.device

        assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        self.coefficient = torch.sqrt(torch.FloatTensor([self.config.n_embd], device=device)) # is this necessary? since its a leaf tensor
        pos = torch.arange(0, t, dtype=torch.long, device=device) # shape (t)
        
        y = self.encoder.drop((self.encoder.wte(input_ids) * self.coefficient) + self.encoder.wpe(pos))

        for layer in self.encoder.h:
            y = layer(y, input_mask)

        y = self.encoder.ln_f(y)
        return y

class DecoderBlock(nn.Module):
    def __init__(self, config: TConfig):
        super().__init__()
        # as said previously, pre-LN is better than post-LN.
        self.ln_1 = LayerNorm(config.n_embd, config.bias)
        self.attn = MultiheadAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, config.bias)
        self.attn_2 = MultiheadAttention(config)
        self.mlp = MLP(config)
        self.drop = nn.Dropout(config.dropout)

    def forward(self, target, encoded_input, input_mask):
        # why an input mask is necessary?
        # in translation tasks, we need to mask out the padding tokens
        
        target = self.ln_1(target)
        # attention is masked by default
        target = target + self.drop(self.attn(target, target, target))
        target = self.ln_2(target)
        encoded_input = self.ln_2(encoded_input)
        y = target + self.drop(self.attn(target, encoded_input, encoded_input, mask=input_mask))
        y = self.drop(y)
        y = y + self.drop(self.mlp(self.ln_2(y)))
        return y

class Decoder(nn.Module):
    def __init__(self, config: TConfig):
        super().__init__()
        self.config = config
        self.decoder = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList([DecoderBlock(config) for _ in range(config.n_layer)]),
            ln_f = LayerNorm(config.n_embd, bias=config.bias),
        ))

    def forward(self, target, encoded_input, input_mask):
        T = target.size(1)
        device = target.device

        assert T <= self.config.block_size, f"Cannot forward sequence of length {T}, block size is only {self.config.block_size}"
        self.coefficient = torch.sqrt(torch.FloatTensor([self.config.n_embd], device=device)) # is this necessary? since its a leaf tensor
        pos = torch.arange(0, T, dtype=torch.long, device=device) # shape (t)
        
        target = self.decoder.drop((self.decoder.wte(target) * self.coefficient) + self.decoder.wpe(pos))

        for layer in self.decoder.h:
            target = layer(target, encoded_input, input_mask)

        target = self.decoder.ln_f(target)
        return target

class Transformer(nn.Module):
    def __init__(self, encoder:Encoder, decoder:Decoder, config:TConfig, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.config = config
        self.device = device
        self.padding_idx = config.padding_idx

        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, device=device)

        self.apply(self._init_weights)

        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))

        # report number of parameters

        print("total params: %.2fM\t\tnon-embedding params: %.2fM" % (self.get_num_params()/1e6,self.get_num_params(non_embedding=True)/1e6,))
    def get_num_params(self, non_embedding=True):
        """
        Return the number of parameters in the model.
        For non-embedding count (default), the position embeddings get subtracted.
        The token embeddings would too, except due to the parameter sharing these
        params are actually used as weights in the final layer, so we include them.
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.encoder.encoder.wpe.weight.numel()
            n_params -= self.decoder.decoder.wpe.weight.numel()
        return n_params

    def forward(self, inputs, targets=None):
        input_mask = (inputs != self.padding_idx).unsqueeze(1).unsqueeze(2)

        #encoder feed through
        encoded_inputs = self.encoder(inputs, input_mask)

        #decoder feed_through
        outputs = self.decoder(targets, encoded_inputs, input_mask)

        if targets is not None:
            # if we are given some desired targets also calculate the loss
            logits = self.lm_head(outputs)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            # inference-time mini-optimization: only forward the lm_head on the very last position
            logits = self.lm_head(outputs[:, [-1], :]) # note: using list [-1] to preserve the time dim
            loss = None

        return logits, loss

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def save_pretrained(self, save_directory):
        #check if directory existso
        if not os.path.exists(save_directory):
            os.makedirs(save_directory)
        #save config
        self.config.save(save_directory)
        # save model
        torch.save(self.state_dict(), os.path.join(save_directory, "pytorch_model.bin"))


    @classmethod
    def from_pretrained(cls, pretrained_model_directory):
        # check if dir exists
        if not os.path.exists(pretrained_model_directory):
            raise Exception('Path %s does not exist' % pretrained_model_directory)
        pass

if __name__ == "__main__":
    print("Testing Transformer Model")
    config = TConfig(block_size = 124, 
                     vocab_size = 412, 
                     n_layer = 12,
                     n_head = 12,
                     n_embd = 48,
                     dropout = 0.0,
                     bias = True,
                     padding_idx = 0)
    encoder = Encoder(config)
    decoder = Decoder(config)
    model = Transformer(encoder, decoder, config, device='cuda')
    input = torch.randint(0, 412, (1, 124))
    target = torch.randint(0, 412, (1, 124))
    output = model(input, target)
    model.save_pretrained("test")
    print(output.size())
    print("Done")

