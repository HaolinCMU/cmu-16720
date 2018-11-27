import numpy as np
from util import *
# do not include any more libraries here!
# do not put any code outside of functions!

# Q 2.1
# initialize b to 0 vector
# b should be a 1D array, not a 2D array with a singleton dimension
# we will do XW + b. 
# X be [Examples, Dimensions]
def initialize_weights(in_size,out_size,params,name=''):

    low = -1.0 * np.sqrt(6.0 / (in_size + out_size))
    high = np.sqrt(6.0 / (in_size + out_size))

    W = np.random.uniform(low, high, (in_size, out_size))
    b = np.zeros(out_size)

    params['W' + name] = W
    params['b' + name] = b


# Q 2.2.1
# x is a matrix
# a sigmoid activation function
def sigmoid(x):
    res = 1.0 / (1.0 + np.exp(-x))
    return res

# Q 2.2.2
def forward(X,params,name='',activation=sigmoid):
    """
    Do a forward pass

    Keyword arguments:
    X -- input vector [Examples x D]
    params -- a dictionary containing parameters
    name -- name of the layer
    activation -- the activation function (default is sigmoid)
    """
    pre_act, post_act = None, None
    # get the layer parameters
    W = params['W' + name]
    b = params['b' + name]

    # your code here
    pre_act = np.dot(X, W) + b
    # print(pre_act)
    post_act = activation(pre_act)

    # store the pre-activation and post-activation values
    # these will be important in backprop
    params['cache_' + name] = (X, pre_act, post_act)

    return post_act

# Q 2.2.2 
# x is [examples,classes]
# softmax should be done for each row
def softmax(x):
    # print('x: ', x)

    res = np.zeros(x.shape)
    c = -1.0 * np.max(x, axis=1)
    # print("x shape: ", x.shape)
    # print("c shape: ", c.shape)
    
    x += c.reshape(-1, 1)
    ex = np.exp(x)
    s = np.sum(ex, axis=1).reshape(-1, 1)
    res = np.divide(ex, s)

    # for i in range(x.shape[0]):
    #     ex = np.exp(x[i])
    #     # print('ex: ', ex)
    #     s = np.sum(ex)
    #     # print('s: ', s)
    #     res[i] = ex / s

    return res

# Q 2.2.3
# compute total loss and accuracy
# y is size [examples,classes]
# probs is size [examples,classes]
def compute_loss_and_acc(y, probs):
    loss, acc = None, None

    loss = -1.0 * np.sum(np.multiply(y, np.log(probs)))
    # print(probs)
    # print(y)
    acc = np.sum(np.equal(np.argmax(y, axis=-1), np.argmax(probs, axis=-1))) / y.shape[0]

    return loss, acc 

# we give this to you
# because you proved it
# it's a function of post_act
def sigmoid_deriv(post_act):
    res = post_act*(1.0-post_act)
    return res

def backwards(delta,params,name='',activation_deriv=sigmoid_deriv):
    """
    Do a backwards pass

    Keyword arguments:
    delta -- errors to backprop
    params -- a dictionary containing parameters
    name -- name of the layer
    activation_deriv -- the derivative of the activation_func
    """
    grad_X, grad_W, grad_b = None, None, None
    # everything you may need for this layer
    W = params['W' + name]
    b = params['b' + name]
    X, pre_act, post_act = params['cache_' + name]
    # your code here
    # do the derivative through activation first
    # then compute the derivative W,b, and X
    # print("X shape: ", X.shape)
    # print("delta shape: ", delta.shape)
    if delta.ndim == 1:
        delta = delta.reshape((1, delta.shape[0]))
    if X.ndim == 1:
        X = X.reshape((1, X.shape[0]))

    delta = delta * activation_deriv(post_act)

    grad_W = np.dot(X.T, delta)
    # grad_W = np.dot(delta, X.T)
    # print("grad_W shape: ", grad_W.shape) # should be out_size * in_size

    grad_X = np.dot(delta, W.T)
    # grad_X = np.dot(W.T, delta.T)
    # print("grad_X shape: ", grad_X.shape)

    grad_b = np.dot(np.ones((1, delta.shape[0])), delta).reshape(-1)
    # grad_b = np.copy(delta)
    # print("grad_b shape: ", grad_b.shape)

    # store the gradients
    params['grad_W' + name] = grad_W
    params['grad_b' + name] = grad_b
    # print("Shape !!!!! ", params['grad_b' + name].shape)
    return grad_X

# Q 2.4
# split x and y into random batches
# return a list of [(batch1_x,batch1_y)...]
def get_random_batches(x,y,batch_size):
    batches = []
    num_examples = x.shape[0]
    num_dimension = x.shape[1]
    num_output = y.shape[1]
    num_batches = int(num_examples / batch_size)

    for batch_id in range(num_batches):
        id_selected = np.random.choice(np.arange(num_examples), size=batch_size, replace=False)
        xi = np.zeros((batch_size, num_dimension))
        yi = np.zeros((batch_size, num_output))

        # for i in range(batch_size):
        #     xi[i] = x[id_selected[i]]
        #     yi[i] = y[id_selected[i]]
        xi[:] = x[id_selected]
        yi[:] = y[id_selected]
        batches.append((xi, yi))

    return batches
