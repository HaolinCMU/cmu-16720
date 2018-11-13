import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
import matplotlib.patches as patches

# write your script here, we recommend the above libraries for making your animation
from LucasKanadeAffine import LucasKanadeAffine
from InverseCompositionAffine import InverseCompositionAffine
from SubtractDominantMotion import SubtractDominantMotion

import cv2

frames = np.load('../data/aerialseq.npy')
num_frames = frames.shape[2]

fig = plt.figure()

for i in range(0, num_frames-1):

    # M = InverseCompositionAffine(frames[:,:,i], frames[:,:,i+1])
    # print(M)
    # warped_img = cv2.warpAffine(frames[:,:,i], M, (frames[:,:,i].shape[1], frames[:,:,i].shape[0]))
    # diff = frames[:,:,i+1] - warped_img

    mask = SubtractDominantMotion(frames[:,:,i], frames[:,:,i+1])
    img = np.copy(frames[:,:,i+1])

    img = np.stack((img, img, img), axis=2) * 255.0
    img[:,:,2] += (mask.astype(np.float32)) * 100.0

    img = np.clip(img, 0, 255).astype(np.uint8)


    plt.imshow(img)

    if i in [30, 60, 90, 120]:
        plt.savefig("aerial"+str(i)+".png")
    plt.pause(0.05)
