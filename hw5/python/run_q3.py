import numpy as np
import scipy.io
from nn import *

train_data = scipy.io.loadmat('../data/nist36_train.mat')
valid_data = scipy.io.loadmat('../data/nist36_valid.mat')
test_data = scipy.io.loadmat('../data/nist36_test.mat')

train_x, train_y = train_data['train_data'], train_data['train_labels']
valid_x, valid_y = valid_data['valid_data'], valid_data['valid_labels']
test_x, test_y = test_data['test_data'], test_data['test_labels']

max_iters = 100
# pick a batch size, learning rate
batch_size = 32
learning_rate = 3e-3
hidden_size = 64

batches = get_random_batches(train_x,train_y,batch_size)
batch_num = len(batches)

params = {}

loss_list = []
acc_list = []
valid_acc_list = []
epoch_list = np.arange(max_iters)

# initialize layers here
initialize_weights(1024,64,params,'layer1')
initialize_weights(64,36,params,'output')


# with default settings, you should get loss < 150 and accuracy > 80%
for itr in range(max_iters):
    total_loss = 0
    total_acc = 0

    for xb,yb in batches:
        # forward
        h1 = forward(xb, params, 'layer1')
        # print(h1)
        probs = forward(h1, params, 'output', softmax)
        # print('probs: ', probs)
        # loss
        # be sure to add loss and accuracy to epoch totals 
        loss, acc = compute_loss_and_acc(yb, probs)
        # print(acc)
        total_loss += loss
        total_acc += acc

        # backward
        # predicted = np.zeros(probs.shape)
        # pred_y = np.argmax(probs, axis=-1) 
        # for i in range(pred_y.shape[0]):
        #         predicted[i, pred_y[i]] = 1.0
        # # predicted[pred_y] = 1.0
        # delta1 = probs - predicted
        # print(delta1)
        delta1 = probs - yb

        delta2 = backwards(delta1, params, 'output', linear_deriv)
        backwards(delta2, params, 'layer1', sigmoid_deriv)
        
        # apply gradient
        params['W' + 'output'] -= learning_rate * params['grad_W' + 'output']
        params['b' + 'output'] -= learning_rate * params['grad_b' + 'output']
        params['W' + 'layer1'] -= learning_rate * params['grad_W' + 'layer1']
        params['b' + 'layer1'] -= learning_rate * params['grad_b' + 'layer1']

    total_loss /= len(batches)
    total_acc /= len(batches)

    loss_list.append(total_loss)
    acc_list.append(total_acc)

    if itr % 2 == 0:
        print("itr: {:02d} \t loss: {:.2f} \t acc : {:.2f}".format(itr,total_loss,total_acc))

    # run on validation set and report accuracy! should be above 75%
    valid_acc = None
    h1 = forward(valid_x, params, 'layer1')
    probs = forward(h1, params, 'output', softmax)
    valid_loss, valid_acc = compute_loss_and_acc(valid_y, probs)
    valid_acc_list.append(valid_acc)
    if itr % 2 == 0:
        print("Validation set accuracy: ", valid_acc)


print('Validation accuracy: ',valid_acc)

import matplotlib.pyplot as plt

plt.figure(1)
ax = plt.gca()
ax.set_xlabel('Epoch')
ax.set_ylabel('Loss (per batch)')
ax.plot(epoch_list, loss_list, color='r', linewidth=2, alpha=1.0, label='loss')
ax.legend()

plt.figure(2)
ax = plt.gca()
ax.set_xlabel('Epoch')
ax.set_ylabel('Accuracy')
ax.plot(epoch_list, acc_list, color='r', linewidth=2, alpha=1.0, label='acc_train')
ax.plot(epoch_list, valid_acc_list, color='b', linewidth=2, alpha=1.0, label='acc_valid')
ax.legend()

plt.show()

if False: # view the data
    for crop in xb:
        import matplotlib.pyplot as plt
        plt.imshow(crop.reshape(32,32).T)
        plt.show()
import pickle
saved_params = {k:v for k,v in params.items() if '_' not in k}
with open('q3_weights.pickle', 'wb') as handle:
    pickle.dump(saved_params, handle, protocol=pickle.HIGHEST_PROTOCOL)

# Q3.1.3
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid

with open('q3_weights.pickle', 'rb') as handle:
   saved_params = pickle.load(handle)

# Learned weights
weights = saved_params['Wlayer1']

fig = plt.figure(3)
grid = ImageGrid(fig, 111, (8,8))
for i in range(64):
    weight_i = weights[:, i].reshape(32, 32)
    grid[i].imshow(weight_i)
plt.show()

# Original weights
initialize_weights(1024, 64, saved_params, 'orig')
weights_orig = saved_params['Worig']

fig = plt.figure(4)
grid = ImageGrid(fig, 111, (8,8))
for i in range(64):
    weight_i = weights_orig[:, i].reshape(32, 32)
    grid[i].imshow(weight_i)
plt.show()



# Q3.1.4
with open('q3_weights.pickle', 'rb') as handle:
   saved_params = pickle.load(handle)

# Construct the confusion matrix
confusion_matrix = np.zeros((train_y.shape[1],train_y.shape[1]))

h1 = forward(test_x, saved_params, 'layer1')
probs = forward(h1, saved_params, 'output', softmax)

ground_truth = np.argmax(test_y, axis=1)
predicted = np.argmax(probs, axis=1)

num_samples = ground_truth.shape[0]

for i in range(num_samples):
    confusion_matrix[ground_truth[i], predicted[i]] += 1
# print(confusion_matrix)


# Visualization of the confusion matrix
import string
plt.imshow(confusion_matrix,interpolation='nearest')
plt.grid(True)
plt.xticks(np.arange(36),string.ascii_uppercase[:26] + ''.join([str(_) for _ in range(10)]))
plt.yticks(np.arange(36),string.ascii_uppercase[:26] + ''.join([str(_) for _ in range(10)]))
plt.show()