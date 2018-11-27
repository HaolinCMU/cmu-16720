import numpy as np
import scipy.io
from nn import *
from collections import Counter

train_data = scipy.io.loadmat('../data/nist36_train.mat')
valid_data = scipy.io.loadmat('../data/nist36_valid.mat')

# we don't need labels now!
train_x = train_data['train_data']
valid_x = valid_data['valid_data']

max_iters = 100
# pick a batch size, learning rate
batch_size = 36 
learning_rate =  3e-5
hidden_size = 32
lr_rate = 20
batches = get_random_batches(train_x,np.ones((train_x.shape[0],1)),batch_size)
batch_num = len(batches)

params = Counter()

# initialize layers here
initialize_weights(1024,32,params,'layer1') # relu
initialize_weights(32,32,params,'hidden') # relu
initialize_weights(32,32,params,'hidden2') # relu
initialize_weights(32,1024,params,'output') # sigmoid

# should look like your previous training loops
for itr in range(max_iters):
    total_loss = 0
    for xb,_ in batches:
        pass
        # training loop can be exactly the same as q2!
        # your loss is now squared error
        # delta is the d/dx of (x-y)^2
        # to implement momentum
        #   just use 'm_'+name variables
        #   to keep a saved value over timestamps
        #   params is a Counter(), which returns a 0 if an element is missing
        #   so you should be able to write your loop without any special conditions

        # forward
        h1 = forward(xb, params, 'layer1', relu)
        h2 = forward(h1, params, 'hidden', relu)
        h3 = forward(h2, params, 'hidden2', relu)
        probs = forward(h3, params, 'output', sigmoid)

        loss = np.sum((probs - xb) * (probs - xb))

        total_loss += loss

        # backward
        delta1 = 2.0 * (probs - xb)

        delta2 = backwards(delta1, params, 'output', sigmoid_deriv)
        delta3 = backwards(delta2, params, 'hidden2', relu_deriv)
        delta4 = backwards(delta3, params, 'hidden', relu_deriv)
        backwards(delta4, params, 'layer1', relu_deriv)
        
        # apply gradient
        params['m_W' + 'output'] = 0.9 *  params['m_W' + 'output'] - learning_rate * params['grad_W' + 'output']
        params['m_b' + 'output'] = 0.9 *  params['m_b' + 'output'] - learning_rate * params['grad_b' + 'output']
        params['W' + 'output'] += params['m_W' + 'output']
        params['b' + 'output'] += params['m_b' + 'output']

        params['m_W' + 'hidden2'] = 0.9 *  params['m_W' + 'hidden2'] - learning_rate * params['grad_W' + 'hidden2']
        params['m_b' + 'hidden2'] = 0.9 *  params['m_b' + 'hidden2'] - learning_rate * params['grad_b' + 'hidden2']
        params['W' + 'hidden2'] += params['m_W' + 'hidden2']
        params['b' + 'hidden2'] += params['m_b' + 'hidden2']

        params['m_W' + 'hidden'] = 0.9 *  params['m_W' + 'hidden'] - learning_rate * params['grad_W' + 'hidden']
        params['m_b' + 'hidden'] = 0.9 *  params['m_b' + 'hidden'] - learning_rate * params['grad_b' + 'hidden']
        params['W' + 'hidden'] += params['m_W' + 'hidden']
        params['b' + 'hidden'] += params['m_b' + 'hidden']

        params['m_W' + 'layer1'] = 0.9 *  params['m_W' + 'layer1'] - learning_rate * params['grad_W' + 'layer1']
        params['m_b' + 'layer1'] = 0.9 *  params['m_b' + 'layer1'] - learning_rate * params['grad_b' + 'layer1']
        params['W' + 'layer1'] += params['m_W' + 'layer1']
        params['b' + 'layer1'] += params['m_b' + 'layer1']

        # print("Momentum: ", np.sum(params['m_' + 'output']))
        # total_acc /= len(batches)

        # loss_list.append(total_loss)
        # acc_list.append(total_acc)

    if itr % 2 == 0:
        print("itr: {:02d} \t loss: {:.2f}".format(itr,total_loss))
    if itr % lr_rate == lr_rate-1:
        learning_rate *= 0.9
# visualize some results
# Q5.3.1
import matplotlib.pyplot as plt
h1 = forward(xb,params,'layer1',relu)
h2 = forward(h1,params,'hidden',relu)
h3 = forward(h2,params,'hidden2',relu)
out = forward(h3,params,'output',sigmoid)
for i in range(5):
    plt.subplot(2,1,1)
    plt.imshow(xb[i].reshape(32,32).T)
    plt.subplot(2,1,2)
    plt.imshow(out[i].reshape(32,32).T)
    plt.show()


from skimage.measure import compare_psnr as psnr
# evaluate PSNR
# Q5.3.2
num_valid = valid_x.shape[0]
psnr_total = 0

h1 = forward(valid_x,params,'layer1',relu)
h2 = forward(h1,params,'hidden',relu)
h3 = forward(h2,params,'hidden2',relu)
out = forward(h3,params,'output',sigmoid)

for i in range(num_valid):
    psnr_total += psnr(valid_x[i], out[i])
psnr_avg = psnr_total / num_valid

print("PSNR Avg: ", psnr_avg)
