import numpy as np


#definee queries, keys and values
quieries = np.array([[1, 0, 1], [0, 1, 1]])
keys = np.array([[1, 0, 1], [1, 1, 0], [0, 1, 1]])
values = np.array([[10, 0], [0, 10], [5, 5]])


#compute attention scores
scores = np.dot(quieries, keys.T)


#apply softmax to normalize scores
def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))  # for numerical stability
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)


attention_weights = softmax(scores)


#comppute weighted sum of values
context = np.dot(attention_weights, values)


print("Attention Weights: \n", attention_weights)
print("Context Vectore: \n", context)

