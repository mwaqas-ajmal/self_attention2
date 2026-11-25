import math
import torch
import torch.nn.functional as F

def attention_forward(q, k, v, mask=None):
    d_k = q.size(-1)
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float("-inf"))
    weights = F.softmax(scores, dim=-1)
    return torch.matmul(weights, v)

# Dummy tensors (batch=1, heads=2, seq_len=4, d_k=8)
q = torch.randn(1, 2, 4, 8)
k = torch.randn(1, 2, 4, 8)
v = torch.randn(1, 2, 4, 8)

output = attention_forward(q, k, v)
print("Output shape:", output.shape)
print(output)
