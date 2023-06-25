import os
import requests
import tiktoken
import numpy as np

input_file_path = os.path.join(os.path.dirname(__file__), "shakespeare.txt")
if not os.path.exists(input_file_path):
    data_url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    with open(input_file_path, "w") as f:
        f.write(requests.get(data_url).text)

with open(input_file_path, "r") as f:
    text = f.read()
n = len(text)

train_data = text[:int(n * 0.9)]
val_data = text[int(n * 0.9):]

enc = tiktoken.get_encoding("gpt2")
train_ids = enc.encode_ordinary(train_data)
val_ids = enc.encode_ordinary(val_data)

train_ids = np.array(train_ids, dtype=np.uint16)
val_ids = np.array(val_ids, dtype=np.uint16)

train_ids.tofile("train.bin")
val_ids.tofile("val.bin")
