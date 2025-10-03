import torch
import torch.nn.functional as F


#definee queries, keys and values
quieries = torch.tensor([[1.0, 0.0, 1.0], [0.0, 1.0, 1.0]])
keys = torch.tensor([[1.0, 0.0, 1.0], [1.0, 1.0, 0.0], [0.0, 1.0, 1.0]])
values = torch.tensor([[10.0, 0.0], [0.0, 10.0], [5.0, 5.0]])


#scores
scores = torch.matmul(quieries, keys.T)


#applying softmax
attention_weights = F.softmax(scores, dim=-1)

#compute weighted sum of values
context = torch.matmul(attention_weights, values)


print("Attention Weights: \n", attention_weights)
print("Context Vectore: \n", context)


import matplotlib.pyplot as plt


plt.matshow(attention_weights)
plt.colorbar()
plt.title("Attention Weights")
plt.show()