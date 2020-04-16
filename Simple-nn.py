import numpy as np 

def sigmoid(x , deri = False):
    if deri == True:
        return x*(1-x)
    return 1/(1 + np.exp(-x))

X = np.array([[0,0,1],
            [0,1,1],
            [1,0,1],
            [1,1,1]])

# x1 = 0
# x2 = 0   For first training example and so on for 4 others ...
# x3 = 1

y = np.array([[0,1,1,0]]).T

np.random.seed(1)

syn0 = 2*np.random.random((3,4)) - 1 # assigning random weights for first layer
syn1 = 2*np.random.random((4,1)) - 1 # assigning random weights for second layer

# 3----4 --- 1

for i in range(150000):
    
    l0 = X
    l1 = sigmoid(np.dot(X,syn0))
    l2 = sigmoid(np.dot(l1 , syn1))

    l2_error = y - l2

    #Bckpropogation

    l2_delta = l2_error*sigmoid(l2,deri=True)
    l1_errors = l2_delta.dot(syn1.T)
    l1_delta = l1_errors*sigmoid(l1 , deri=True)

    syn1 += l1.T.dot(l2_delta)
    syn0 += l0.T.dot(l1_delta)

print(l2)    
