import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
import matplotlib.patches as patches

# write your script here, we recommend the above libraries for making your animation
from LucasKanade import LucasKanade

frames = np.load('../data/carseq.npy')
num_frames = frames.shape[2]

rect_init = np.array([[59], [116], [145], [151]]).astype(np.float32)
rect = np.copy(rect_init)
rectList = np.empty((0, 4))
rectList = np.append(rectList, rect.reshape((1,4)), axis=0)

width = rect[2] - rect[0]
height = rect[3] - rect[1]

sigma = 1

fig = plt.figure(1)
ax = fig.add_subplot(111)
ax.set_title("My Title")

# Run LK tracking with template correction in real-time
print("Running LK tracking (with template correction) in real-time...")
for i in range(0, num_frames-1):
    print("frame: ", i)
    p = LucasKanade(frames[:,:,i], frames[:,:,i+1], rect, np.zeros(2))
    #print("p: ", p)
    p0 = np.array((rect[1] + p[0] - rect_init[1], rect[0] + p[1] - rect_init[0])).reshape(2)
    #print("p0: ", p0)
    pn = LucasKanade(frames[:,:,0], frames[:,:,i+1], rect_init, p0)
    #print("pn: ", pn)

    p_norm = np.linalg.norm(pn - p0)
    #print('p_norm: ', p_norm)
    if p_norm < sigma:
        rect[0] += p[1]
        rect[1] += p[0]
        rect[2] = rect[0] + width
        rect[3] = rect[1] + height
    else:
        rect[0] = rect_init[0] + pn[1]
        rect[1] = rect_init[1] + pn[0]
        rect[2] = rect[0] + width
        rect[3] = rect[1] + height

    rectList = np.append(rectList, rect.reshape((1,4)), axis=0)
    img = frames[:,:,i].copy()

    ax.add_patch(patches.Rectangle((rect[0], rect[1]), rect[2]-rect[0]+1, rect[3]-rect[1]+1, linewidth=2, edgecolor='green', fill=False))
    plt.imshow(img, cmap='gray')
    plt.pause(0.01)
    ax.clear()
np.save('carseqrects-wcrt.npy', rectList)

rectListOrig = np.load('carseqrects.npy')
rectList = np.load('carseqrects-wcrt.npy')

# Playback frames with rectangles
print("Playing back...")
for i in range(0, num_frames-1):
    print("frame: ", i)
    img = frames[:,:,i].copy()

    rect = rectList[i, :]
    rect0 = rectListOrig[i, :]

    ax.add_patch(patches.Rectangle((rect[0], rect[1]), rect[2]-rect[0]+1, rect[3]-rect[1]+1, linewidth=2, edgecolor='green', fill=False))
    ax.add_patch(patches.Rectangle((rect0[0], rect0[1]), rect0[2]-rect0[0]+1, rect0[3]-rect0[1]+1, linewidth=2, edgecolor='red', fill=False))

    plt.imshow(img, cmap='gray')
    if i in [1, 100, 200, 300, 400]:
        plt.savefig("carWithCorrection"+str(i)+".png")
    plt.pause(0.01)
    ax.clear()
