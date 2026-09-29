import numpy as np

def sigmoid(z):
    return 1 / (1+np.exp(-z))

np.random.seed(0)
X = np.random.randn(100,2)
true_w = np.array([2.0, -1.0])
y = (X @ true_w > 0).astype(float)
#print(y)

w = np.zeros(2)
b = 0.0
z = X @ w + b
p = sigmoid(z)

#print(p)


