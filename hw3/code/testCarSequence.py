import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
import matplotlib.patches as patches
# write your script here, we recommend the above libraries for making your animation

from LucasKanade import LucasKanade

frames = np.load('../data/carseq.npy')
num_frames = frames.shape[2]

rect = np.array([[59], [116], [145], [151]]).astype(np.float32)
width = rect[2] - rect[0]
height = rect[3] - rect[1]

rectList = np.empty((0, 4))
rectList = np.append(rectList, rect.reshape((1,4)), axis=0)

fig = plt.figure(1)
ax = fig.add_subplot(111)
ax.set_title("LucasKanade Tracking")

# Run LK tracking in real-time
print("Running LK tracking in real-time...")
for i in range(0, num_frames-1):
    print("frame: ", i)
    p = LucasKanade(frames[:,:,i], frames[:,:,i+1], rect)
    rect[0] += p[1]
    rect[1] += p[0]
    rect[2] = rect[0] + width
    rect[3] = rect[1] + height

    rectList = np.append(rectList, rect.reshape((1,4)), axis=0)

    ax.add_patch(patches.Rectangle((rect[0], rect[1]), rect[2]-rect[0]+1, rect[3]-rect[1]+1, linewidth=2, edgecolor='red', fill=False))
    plt.imshow(frames[:,:,i+1], cmap='gray')
    plt.pause(0.01)
    ax.clear()
np.save('carseqrects.npy', rectList)

# Playback frames with rectangles
rectList = np.load('carseqrects.npy')
print("Playing back frames...")
for i in range(0, num_frames-1):
    print("frame: ", i)
    img = frames[:,:,i].copy()
    rect = rectList[i, :]

    ax.add_patch(patches.Rectangle((rect[0], rect[1]), rect[2]-rect[0]+1, rect[3]-rect[1]+1, linewidth=2, edgecolor='red', fill=False))
    plt.imshow(img, cmap='gray')
    if i in [1, 100, 200, 300, 400]:
        plt.savefig("car"+str(i)+".png")
    plt.pause(0.01)
    ax.clear()
