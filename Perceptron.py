import numpy as np 

def sigmoid(x):
    return 1/(1 + np.exp(-x))

def sigmoid_derivate(x):
    return x*(1-x)

x = np.array([[0,0,1],
             [1,1,1],
             [1,0,1],
             [0,1,1]])

# x1 = 0
# x2 = 0   For first training example and so on for 4 others ...
# x3 = 1

y = np.array([[0,1,1,0]]).T #taking transpose

np.random.seed(1)

epoch = 15000

syn0 = 2*np.random.random((3,1)) - 1 # assigning random weights


#Training Neural network 

for i in range(epoch): 

    l0 = x  # taking input layer

    output = sigmoid(np.dot(l0,syn0)) # finding output using sigmoid function 

    error = y-output # calculating error

    delta1 = error * sigmoid_derivate(output)

    syn0 += np.dot(l0.T , delta1) # update weights

print(output)