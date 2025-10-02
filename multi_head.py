import numpy as np

#define query, key and value
def generate_data(seq_len, embed_dim):
    np.random.seed(42)
    return np.random.rand(seq_len, embed_dim)

sequence_length =4
embedding_dim = 3
query = generate_data(sequence_length, embedding_dim)
key = generate_data(sequence_length, embedding_dim)
value = generate_data(sequence_length, embedding_dim)

#compute attention score
scores = np.dot(query,key.T) / np.sqrt(embedding_dim)


#apply softmax to normalize scores
def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return exp_x / exp_x.sum(axis=-1, keepdims=True)

attention_weights = softmax(scores)

#compute the context
context = np.dot(attention_weights, value)

#print("Attention Weights: \n", attention_weights)
#print("Context Vector:\n", context)



import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        assert embed_dim  % num_heads == 0,  "Embedding dimension must be divisible by number of heads"
        
        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        self.out = nn.Linear(embed_dim, embed_dim)
        
        
    def forward(self, x):
        batch_size = x.size(0)
        
        #linear projections
        q = self.query(x).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1,2)
        k = self.query(x).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1,2)
        v = self.query(x).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1,2)
        
        
        #compute attention scores
        scores = torch.matmul(q, k.transpose(-2, -2)) / np.sqrt(self.head_dim)
        attention_weights = F.softmax(scores, dim=-1)
        
        #compute context
        context = torch.matmul(attention_weights, v).transpose(1,2).contiguous().view(batch_size, -1, self.embed_dim)
        return self.out(context), attention_weights
    
    #sample input
seq_len, embed_dim = 4,8
x = torch.rand(1, seq_len, embed_dim)

#instantiate and test
mha = MultiHeadAttention(embed_dim, num_heads=2)
context, attn_weights = mha(x)
print("Attention Weights: \n", attn_weights)
print("Context Vector:\n", context)

import matplotlib.pyplot as plt

#visualize
plt.matshow(attn_weights[0, 0].detach().numpy(), cmap="viridis")
plt.colorbar()
plt.title("Attention Weights (Head 1)")
plt.show()
