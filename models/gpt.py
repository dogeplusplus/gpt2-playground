import math
import torch
import inspect
import torch.nn as nn
import torch.nn.functional as F

from einops import repeat, rearrange
from dataclasses import dataclass


class LayerNorm(nn.Module):
    def __init__(self, dim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim)) if bias else None

    def forward(self, x):
        return F.layer_norm(x, self.weight.shape, self.weight, self.bias, 1e-5)


class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout

        self.flash = hasattr(torch.nn.functional, "scaled_dot_product_attention")
        if not self.flash:
            bias = repeat(torch.tril(torch.ones(config.block_size, config.block_size)), "i j -> 1 1 i j")
            self.register_buffer("bias", bias)

    def forward(self, x):
        T = x.size(1)
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)

        q = rearrange(q, "b t (h d) -> b h t d", h=self.n_head)
        k = rearrange(k, "b t (h d) -> b h t d", h=self.n_head)
        v = rearrange(v, "b t (h d) -> b h t d", h=self.n_head)

        if self.flash:
            y = F.scaled_dot_product_attention(
                q, k, v, attn_mask=None, dropout_p=self.dropout if self.training else 0.0)
        else:
            att = q @ rearrange(k, "b h t d -> b h d t") / math.sqrt(k.size(-1))
            att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf"))
            att = F.softmax(att, dim=-1)
            y = att @ v

        y = rearrange(y, "b h t d -> b t (h d)")
        y = self.resid_dropout(self.c_proj(y))
        return y


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class Block(nn.Module):
    def __init__(self, config: dict, enc_embd: int = None, max_seq_length: int = 1024):
        super().__init__()
        self.ln1 = LayerNorm(config.n_embd, config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln2 = LayerNorm(config.n_embd, config.bias)
        self.mlp = MLP(config)
        if enc_embd is not None:
            self.cross_attn = nn.MultiheadAttention(
                config.n_embd,
                config.n_head,
                dropout=config.dropout,
                kdim=enc_embd,
                vdim=enc_embd,
                batch_first=True,
            )
            # Decoder can only attend to current state of board and past states in cross attention
            self.register_buffer("attn_mask", torch.tril(torch.ones(max_seq_length, max_seq_length)))

    def forward(self, x, enc_out=None):
        x = x + self.attn(self.ln1(x))
        if enc_out is not None:
            seq_len = x.size(1)
            attn_mask = self.attn_mask[:seq_len, :seq_len]
            c = self.cross_attn(self.ln2(x), enc_out, enc_out, attn_mask=attn_mask)[0]
            x = x + self.mlp(c)
        else:
            x = x + self.mlp(self.ln2(x))
        return x


@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 1024
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = True
    encoder_decoder: bool = False


class GPT(nn.Module):
    def __init__(self, config, encoder=None):
        super().__init__()
        self.config = config

        self.transformer_encoder = encoder
        if encoder is not None:
            self.encoder_decoder = True
            enc_embd = encoder.config.n_embd
        else:
            self.encoder_decoder = False
            enc_embd = None

        self.transformer_decoder = nn.ModuleDict(dict(
            wte=nn.Embedding(config.vocab_size, config.n_embd),
            wpe=nn.Embedding(config.block_size, config.n_embd),
            drop=nn.Dropout(config.dropout),
            h=nn.ModuleList([Block(config, enc_embd, config.block_size) for _ in range(config.n_layer)]),
            ln_f=LayerNorm(config.n_embd, config.bias),
        ))

        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer_decoder.wte.weight = self.lm_head.weight

        self.apply(self._init_weights)

        for pn, p in self.named_parameters():
            if pn.endswith("c_proj.weight"):
                nn.init.normal_(p, mean=0., std=0.02/math.sqrt(2 * config.n_layer))

        print("number of parameters: ", sum(p.numel() for p in self.parameters() if p.requires_grad))

    def get_num_params(self, non_embedding=True):
        n_params = sum(p.numel() for p in self.parameters())

        if non_embedding:
            n_params -= self.transformer_decoder.wpe.weight.numel()
        return n_params

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0., std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0., std=0.02)

    def forward(self, idx, targets=None, board_states=None):
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.block_size, \
            f"Cannot forward, model block size is {self.config.block_size}, but got sequence of length {t}"
        pos = torch.arange(0, t, dtype=torch.long, device=device)

        # forward the gpt model
        tok_emb = self.transformer_decoder.wte(idx)
        pos_emb = self.transformer_decoder.wpe(pos)
        x = self.transformer_decoder.drop(tok_emb + pos_emb)

        enc_out = None
        if self.encoder_decoder and board_states is not None:
            enc_out = self.transformer_encoder(board_states)

        for block in self.transformer_decoder.h:
            x = block(x, enc_out)

        x = self.transformer_decoder.ln_f(x)

        if targets is not None:
            targets = torch.where(targets == 0, -1, targets)
            logits = self.lm_head(x)
            loss = F.cross_entropy(rearrange(logits, "b t v -> (b t) v"),
                                   rearrange(targets, "b t -> (b t)"), ignore_index=-1)
        else:
            logits = self.lm_head(x[:, [-1], :])
            loss = None

        return logits, loss

    def crop_block_size(self, block_size):
        assert block_size <= self.config.block_size
        self.config.block_size = block_size
        self.transformer_decoder.wpe = nn.Parameter(self.transformer_decoder.wpi.weight[:block_size])
        for block in self.transformer_decoder.h:
            if hasattr(block.attn, "bias"):
                block.attn.bias = block.attn.bias[:, :, :block_size, :block_size]

    @classmethod
    def from_pretrained(cls, model_type, override_args=None):
        assert model_type in {"gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"}
        override_args = override_args or {}

        assert all(k == "dropout" for k in override_args)
        from transformers import GPT2LMHeadModel

        print("loading gpt2 model...")
        config_args = {
            "gpt2": dict(n_layer=12, n_head=12, n_embd=768),
            "gpt2-medium": dict(n_layer=24, n_head=16, n_embd=1024),
            "gpt2-large": dict(n_layer=36, n_head=20, n_embd=1280),
            "gpt2-xl": dict(n_layer=48, n_head=25, n_embd=1600),
        }[model_type]

        config_args["vocab_size"] = 50257
        config_args["block_size"] = 1024
        config_args["bias"] = True

        if "dropout" in override_args:
            config_args["dropout"] = override_args["dropout"]

        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()

        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith(".attn.bias")]

        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith(".attn.masked_bias")]
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith(".attn.bias")]
        transposed = ["attn.c_attn.weight", "attn.c_proj.weight", "mlp.c_fc.weight", "mlp.c_proj.weight"]

        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} vs {len(sd_keys)}"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].T)
            else:
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        param_dict = {pn: p for pn, p in self.named_parameters() if p.requires_grad}

        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": nodecay_params, "weight_decay": 0.},
        ]

        fused_available = "fused" in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == "cuda"
        extra_args = dict(fused=True) if use_fused else {}
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)

        return optimizer

    def estimate_mfu(self, fwdbwd_per_iter, dt):
        N = self.get_num_params()
        cfg = self.config
        L, H, Q, T = cfg.n_layer, cfg.n_head, cfg.n_embd // cfg.n_head, cfg.block_size
        flops_per_token = 6*N + 12*L*H*Q*T
        flops_per_fwdbwd = flops_per_token * T
        flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter
        flops_achieved = flops_per_iter / dt
        flops_promised = 312e12
        mfu = flops_achieved / flops_promised

        return mfu

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1., top_k=None):
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature

            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("inf")
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)

        return idx


@dataclass
class BoardEncoderConfig:
    n_embd: int = 768
    n_head: int = 12
    board_size: int = 9
    n_layers: int = 2


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=512):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_parameter('pe', nn.Parameter(pe, requires_grad=False))

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class BoardEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.proj = nn.Linear(3 * config.board_size * config.board_size, config.n_embd)
        self.norm = nn.LayerNorm(config.n_embd)
        # Dropout set to 0 for now due to some tracing error with flash attention when compiling the model?
        self.encoder = nn.ModuleList([
            nn.TransformerEncoderLayer(config.n_embd, config.n_head, batch_first=True, dropout=0)
            for _ in range(config.n_layers)
        ])
        self.pos_encoder = PositionalEncoding(config.n_embd)

    def forward(self, x):
        x = rearrange(x, "b t c h w -> b t (c h w)")
        x = self.proj(x)
        x = self.norm(x)
        x = self.pos_encoder(x)
        for layer in self.encoder:
            x = layer(x)
        return x
