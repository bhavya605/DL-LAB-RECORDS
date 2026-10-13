import numpy as np

# Single Perceptron AND Gate
X = np.array([[0,0], [0,1], [1,0], [1,1]])   
y = np.array([0, 0, 0, 1])                   
w = np.random.rand(2)
b = np.random.rand(1)
lr = 0.1 

for epoch in range(20):                               #training
    for i in range(len(X)):
        linear_op = np.dot(X[i], w) + b               #linear combination
        pred = 1 if linear_op >= 0.5 else 0           #step activation 
        er = y[i] - pred                              #update 
        w += lr * er * X[i]
        b += lr * er
print("Final weights for Single Perceptron AND Gate:", w)                            
print("Final bias for Single Perceptron AND Gate:", b)

for i in range(len(X)):                               #testing
    op = 1 if np.dot(X[i], w) + b >= 0.5 else 0
    print(f"Input: {X[i]} -> Predicted: {op}, Expected: {y[i]}")


# Feedforward Neural Network (FFNN) for XOR/AND operation
X = np.array([[0,0],[0,1],[1,0],[1,1]])
y = np.array([[0],[1],[1],[0]])

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

np.random.seed(42)                                     #initializing weights
ip_lyr_nns = 2
hddn_nns = 2
op_nns = 1

W1 = np.random.uniform(size=(ip_lyr_nns, hddn_nns))
W2 = np.random.uniform(size=(hddn_nns, op_nns))
b1 = np.random.uniform(size=(1, hddn_nns))
b2 = np.random.uniform(size=(1, op_nns))
lr = 0.5

for epoch in range(10000):                             #training
    hddn_ip = np.dot(X, W1) + b1                       #forward pass 
    hddn_op = sigmoid(hddn_ip)
    fnl_ip = np.dot(hddn_op, W2) + b2
    fnl_op = sigmoid(fnl_ip)
    
    er = y - fnl_op                                    #backpropagation
    d_op = er * sigmoid_derivative(fnl_op)
    er_hddn = d_op.dot(W2.T)
    d_hddn = er_hddn * sigmoid_derivative(hddn_op)

    W2 += hddn_op.T.dot(d_op) * lr                     #updating weights
    b2 += np.sum(d_op, axis=0, keepdims=True) * lr
    W1 += X.T.dot(d_hddn) * lr
    b1 += np.sum(d_hddn, axis=0, keepdims=True) * lr

print("Final Output after training the FFNN model for XOR:")
print(fnl_op.round())


# Multilayer Perceptron (MLP) for a simple dataset
X = np.random.rand(200, 2)                              #generating dataset (points around line y = x)
y = np.array([[1] if p[1] > p[0] else [0] for p in X])  #label: 1 if y>x else 0

def sigmoid(x):
    return 1 / (1 + np.exp(-x))
def sigmoid_derivative(x):
    return x * (1 - x)

ip_nns = 2                                               #network architecture
hddn_nns = 4
op_nns = 1

W1 = np.random.uniform(size=(ip_nns, hddn_nns))
W2 = np.random.uniform(size=(hddn_nns, op_nns))
b1 = np.random.uniform(size=(1, hddn_nns))
b2 = np.random.uniform(size=(1, op_nns))
lr = 0.1
epochs = 5000

for epoch in range(epochs):                              #training
    hddn_ip = np.dot(X, W1) + b1                         #forward pass
    hddn_op = sigmoid(hddn_ip)
    fnl_ip = np.dot(hddn_op, W2) + b2
    fnl_op = sigmoid(fnl_ip)

    er = y - fnl_op                                      #backpropagation
    d_op = er * sigmoid_derivative(fnl_op)
    er_hddn = d_op.dot(W2.T)
    d_hddn = er_hddn * sigmoid_derivative(hddn_op)

    W2 += hddn_op.T.dot(d_op) * lr                       #update weights
    b2 += np.sum(d_op, axis=0, keepdims=True) * lr
    W1 += X.T.dot(d_hddn) * lr
    b1 += np.sum(d_hddn, axis=0, keepdims=True) * lr

preds = (fnl_op > 0.5).astype(int)                       #testing accuracy
acc = np.mean(preds == y)
print("Training Accuracy:", acc)
