import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches

import skimage
import skimage.measure
import skimage.color
import skimage.restoration
import skimage.io
import skimage.filters
import skimage.morphology
import skimage.segmentation

from nn import *
from q4 import *
# do not include any more libraries here!
# no opencv, no sklearn, etc!
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)


im1 = skimage.img_as_float(skimage.io.imread('../images/01_list.jpg'))
im2 = skimage.img_as_float(skimage.io.imread('../images/02_letters.jpg'))
im3 = skimage.img_as_float(skimage.io.imread('../images/03_haiku.jpg'))
im4 = skimage.img_as_float(skimage.io.imread('../images/04_deep.jpg'))

bboxes, bw = findLetters(im1)

plt.imshow(bw)
for bbox in bboxes:
    minr, minc, maxr, maxc = bbox
    rect = matplotlib.patches.Rectangle((minc, minr), maxc - minc, maxr - minr,
                            fill=False, edgecolor='red', linewidth=1)
    plt.gca().add_patch(rect)
plt.show()
# find the rows using..RANSAC, counting, clustering, etc.

# crop the bounding boxes
# note.. before you flatten, transpose the image (that's how the dataset is!)
# consider doing a square crop, and even using np.pad() to get your images looking more like the dataset

# load the weights
# run the crops through your neural network and print them out
import pickle
import string
letters = np.array([_ for _ in string.ascii_uppercase[:26]] + [str(_) for _ in range(10)])
params = pickle.load(open('q3_weights.pickle','rb'))
    
