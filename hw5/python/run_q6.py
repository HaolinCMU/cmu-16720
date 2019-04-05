import numpy as np
import scipy.io
from nn import *
import matplotlib.pyplot as plt
from skimage.measure import compare_psnr as psnr

train_data = scipy.io.loadmat('../data/nist36_train.mat')
valid_data = scipy.io.loadmat('../data/nist36_valid.mat')

# we don't need labels now!
train_x = train_data['train_data']
valid_x = valid_data['valid_data']

dim = 32
# do PCA

mean = np.mean(train_x, axis=0)
train_x -= mean

U, S, V = np.linalg.svd(np.dot(train_x.T, train_x))

projection = U[:, :dim]
print("Shape of projection matrix: ", projection.shape)

# rebuild a low-rank version
lrank = np.dot(train_x, projection)

# rebuild it
recon = np.dot(lrank, projection.T)

for i in range(5):
    plt.subplot(2,1,1)
    plt.imshow(train_x[i].reshape(32,32).T)
    plt.subplot(2,1,2)
    plt.imshow(recon[i].reshape(32,32).T)
    plt.show()

# build valid dataset
valid_x -= mean
recon_valid = np.dot(np.dot(valid_x, projection), projection.T)
recon_valid += mean
valid_x += mean

# Plot reconsturcted pictures from 5 classes
valid_y = valid_data['valid_labels']
counter = 0

class_id = np.random.choice(np.arange(36), size=5, replace=False)
# print(class_id)
for id in class_id:
    counter = 0
    for i in range(valid_x.shape[0]):
        if valid_y[i][id] == 1:
            counter += 1
            plt.subplot(2,1,1)
            plt.imshow(valid_x[i].reshape(32,32).T)
            plt.subplot(2,1,2)
            plt.imshow(recon_valid[i].reshape(32,32).T)
            plt.show()
            if counter == 2:
                break

total = []
for pred,gt in zip(recon_valid,valid_x):
    total.append(psnr(gt,pred))
print(np.array(total).mean())