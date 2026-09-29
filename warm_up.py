import numpy as np

a = np.array([1,2,3])
b = np.array([[1,2,3],
             [4,5,6]])

print(a.shape)
print(b.shape)

'''
warm up questions
'''

X = np.arange(6).reshape(3,2)
print(X.shape)
w = np.array([1.0, 2.0])
print(w.shape)
F = X @ w
print(F.shape)

'''
shape of X.T is 2,3
shape of X.T @ X is 2,2
X + np.array([10,20]) work because of broadcasting and it shape matches (2,)
X + np.array([10,20,30]) doesn't work because when you broadcast it is too big (3,)
X.sum(axis=0) has a shape of (2,)
X.sum(axis=1) has a shape of (3,)
axis=0 gives the total of each feature column
'''

print(X.mean(axis=0))
centered = X - X.mean(axis=0)
print(centered)
print(centered.mean(axis=0))

def f(w):
    return (2*w - 4) ** 2

w=1.0
h=0.0001

slope=(f(w+h)-f(w)) / h
print(slope)

'''
1. Code will print 6.0001
2. Code will print 12.0006 
3. Code will print -7.9995
Its not the code you suggested because it doesn't have the exponential value in it
'''
