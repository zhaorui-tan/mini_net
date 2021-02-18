from sklearn.datasets import load_boston
import matplotlib.pyplot as plt
from node import *
from ult import *


# Load data
data = load_boston()
X_ = data['data']
y_ = data['target']

# Normalize data
X_ = (X_ - np.mean(X_, axis=0)) / np.std(X_, axis=0)

# print('RR',np.random.random(len(X_[0])))
w1_, b1_ = np.random.random(len(X_[0])), np.random.normal()
w2_, b2_ = np.random.random(len(X_[0])), np.random.normal()
w3_, b3_ = np.random.random(len(X_[0])), np.random.normal()

X, y = Placeholder(name='X', is_trainable=False), Placeholder(name='y', is_trainable=False)
w1, b1 = Placeholder(name='w1'), Placeholder(name='b1')
w2, b2 = Placeholder(name='w2'), Placeholder(name='b2')

# build model
output1 = Linear(X, w1, b1, name='linear-01')
output2 = Sigmoid(output1, name='activation')

# output2 = Relu(output1, name='activation')
y_hat = Linear(output2, w2, b2, name='y_hat')
cost = L2_LOSS(y, y_hat, name='cost')

feed_dict = {
    X: X_,
    y: y_,
    w1: w1_,
    w2: w2_,
    b1: b1_,
    b2: b2_,
}

graph_sort = topological_sort_feed_dict(feed_dict)
epoch = 1000
batch_num = len(X_)
learning_rate = 1e-4

losses = []
for e in range(epoch):
    loss = 0

    # for b in range(batch_num):
    for b in range(10):
        index = np.random.choice(range(len(X_)))
        X.value = X_[index]
        y.value = y_[index]
        forward_and_backward(graph_sort, monitor=False)
        optimize(graph_sort, learning_rate)
        loss += cost.value
    losses.append(loss / batch_num)

plt.plot(losses)
plt.show()