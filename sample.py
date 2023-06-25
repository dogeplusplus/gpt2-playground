import torch
import tiktoken

from train import LitGPT

from model import GPT, GPTConfig

n_layer = 12
n_head = 12
n_embd = 768
bias = True
dropout = 0.0
vocab_size = 50304
accumulation_steps = 5 * 8
batch_size = 16
block_size = 1024

learning_rate = 6e-4
beta1 = 0.9
beta2 = 0.95
weight_decay = 0.1

model_args = dict(
    n_layer=n_layer,
    n_head=n_head,
    n_embd=n_embd,
    block_size=block_size,
    bias=bias,
    vocab_size=vocab_size,
    dropout=dropout,
)

gpt_config = GPTConfig(**model_args)
model = GPT(gpt_config)
gpt_model = LitGPT(
    model,
    learning_rate,
    beta1,
    beta2,
    weight_decay,
)
gpt_model = torch.compile(gpt_model)
device = "cuda" if torch.cuda.is_available() else "cpu"
model = LitGPT.load_from_checkpoint("checkpoints/gpt-epoch=01-val_loss=6.73.ckpt").model.to(device)

start = "\n"
encoder = tiktoken.get_encoding("gpt2")
start_ids = encoder.encode(start)
x = (torch.tensor(start_ids, dtype=torch.long).unsqueeze(0)).to(device)

num_samples = 10
max_new_tokens = 500
with torch.no_grad():
    for i in range(num_samples):
        y = model.generate(x, max_new_tokens=max_new_tokens)
        print(encoder.decode(y[0].tolist()))
