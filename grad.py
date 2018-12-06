import numpy as np
import matplotlib.pyplot as plt

# Helper functions
def sigmoid(z):
    s = 1/(1+np.exp(-z))
    return s

def relu(z):
    s = np.maximum(0,z)
    return s

def relu_backward(dA, cache):
    
    Z = cache
    dZ = np.array(dA, copy=True) # just converting dz to a correct object.
    
    # When z <= 0, you should set dz to 0 as well. 
    dZ[Z <= 0] = 0
    
    assert (dZ.shape == Z.shape)
    
    return dZ

def sigmoid_backward(dA, cache):
    
    Z = cache
    
    s = 1/(1+np.exp(-Z))
    dZ = dA * s * (1-s)
    
    assert (dZ.shape == Z.shape)
    
    return dZ


# Define cost function
def cost_func(AL,Y):
    m = AL.shape[1]
    Y.shape = AL.shape # Just make sure the two vectors shape are aligned
    
    cost = -1./m * np.sum(np.multiply(Y,np.log(AL)) + np.multiply((1-Y),np.log(1-AL)))
    cost = np.squeeze(cost) # Make sure the cost is a scalar
    
    return cost

# Linear computation
def fwd_linear(A_prev,W,b):
    assert (A_prev.shape[0]==W.shape[1])
    assert (W.shape[0]==b.shape[0])
    Z = np.dot(W,A_prev)+b
    cache = (A_prev, W, b)
    return Z, cache

# Linear activation
def fwd_activation(A_prev, W, b, activation_func):
    Z, linear_cache = fwd_linear(A_prev, W, b)
    
    
    if activation_func == "sigmoid":
        A = sigmoid(Z)
    elif activation_func == "relu":
        A = relu(Z)
        
    cache = (linear_cache, Z)
        
    return A, cache

# Forward propagation over the network
def fwd_run(X, parameters):
    L = len(parameters)//2
    m = X.shape[1]
    caches = []
    A = X
    
    # Run the L-1 first activation with relu
    for l in range(1,L):
        A, cache = fwd_activation(A,
                                 W = parameters['W'+str(l)],
                                 b = parameters['b'+str(l)],
                                 activation_func = "relu")
        caches.append(cache)
        
    # Run the L-th activation
    AL, cache = fwd_activation(A,
                       W = parameters['W'+str(L)],
                       b = parameters['b'+str(L)],
                       activation_func = "sigmoid")
    caches.append(cache)
    
    
    return AL, caches
        
# Gradient computation
def back_linear(dZ, linear_cache):
    # Get back values from cache
    A_prev, W, b = linear_cache
    
    # Get the value of m
    m = A_prev.shape[1]
    
    dW = 1./m * np.dot(dZ, A_prev.T)
    db = 1./m * np.sum(dZ, axis=1, keepdims=True)
    dA_prev = np.dot(W.T, dZ)
    
    return dA_prev, dW, db



# Backward activation
def back_activation(dA, cache, activation_func):
    linear_cache, Z = cache # Unpack the cache to get ((A_prev, W, b), Z)
    
    if activation_func == "sigmoid":
        dZ_prev = sigmoid_backward(dA, Z)
        dA_prev, dW, db = back_linear(dZ_prev, linear_cache)
    elif activation_func == "relu":
        dZ_prev = relu_backward(dA, Z)
        dA_prev, dW, db = back_linear(dZ_prev, linear_cache)
        
        
    return dA_prev, dW, db

# Gradient descent
def back_run(AL, Y, caches):
    grads = {}
    L = len(caches)
    Y.shape = AL.shape
    
    dAL = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))
    
    # Run the first pass with the sigmoid
    current_cache = caches[L-1]
    grads['dA'+str(L)], grads['dW'+str(L)] , grads['db'+str(L)] = back_activation(dAL, current_cache, "sigmoid")
    
    # Run the L-1 passes with relu
    for l in reversed(range(L-1)):
        current_cache = caches[l]
        grads['dA'+str(l+1)], grads['dW'+str(l+1)], grads['db'+str(l+1)] = back_activation(grads['dA'+str(l+2)], current_cache, "relu")
    
    return grads

# Parameters update
def update_parameters(parameters, grads, alpha):
    L = len(parameters)//2
    for l in range(1,L+1):
        parameters['W'+str(l)] -= alpha * grads['dW'+str(l)]
        parameters['b'+str(l)] -= alpha * grads['db'+str(l)]
        
    return parameters

# Parameters initialization
def initialize_parameters(X, layer_dims):
    layer_dims.insert(0,X.shape[0])
    L = len(layer_dims)
    params = {}
    for l in range(1,L):
        params['W'+str(l)] = np.random.randn(layer_dims[l], layer_dims[l-1]) * np.sqrt(2/layer_dims[l-1])
        params['b'+str(l)] = np.zeros((layer_dims[l], 1))
        
    return params

# Parameters model -> return what we need
def model(X,Y, layer_dims, num_iter, alpha=0.01):
    # Create a costs list to measure it the algorithm is doing well
    costs = []
    
    #Initialize the parameters
    parameters = initialize_parameters(X, layer_dims)
    
    # Iterate over num_iter examples:
    for i in range(num_iter):
    
        #Run the forward propagation
        AL, caches = fwd_run(X, parameters)
        
        #Compute the cost and store it in costs
        if i%10==0:
            cost = cost_func(AL,Y)
            costs.append(cost)
        
        #Run the back-prop
        grads = back_run(AL, Y, caches)
        
        #Update the parameters
        parameters = update_parameters(parameters, grads, alpha)
        
    return {"parameters": parameters, 
            "costs": costs, 
            "grads": grads}

# Prediction function
def predict(X,Y, parameters, prob_threshold=0.5):
    accuracy = 0
    precision = 0
    recall = 0
    f1_score = 0
    AL, _ = fwd_run(X, parameters)
    Y.shape = AL.shape # Make sure they have the same shape
    
    # Transform the probabilities into an assertion O or 1
    AL[AL<=prob_threshold] = 0
    AL[AL>prob_threshold] = 1
    
    # Compute the difference between AL and Y and get the accuracy
    accuracy = np.sum(AL==Y)/Y.shape[1]
    
    # Compute precision, recall and F1-score
    L = Y.shape[1]
    TP = np.zeros((1,L))
    for l in range(L):
        if ((AL[0,l]==1) & (Y[0,l]==1)):
            TP[0,l]=1
    
    if ((np.sum(AL)!=0) & (np.sum(Y)!=0)):
        precision = np.sum(TP)/np.sum(AL)
        recall = np.sum(TP)/np.sum(Y)
        f1_score = 2 * (precision*recall)/(precision+recall)
        
    scores = {"accuracy": accuracy,
             "precision": precision,
             "recall": recall,
             "f1 score": f1_score}
        
    return scores