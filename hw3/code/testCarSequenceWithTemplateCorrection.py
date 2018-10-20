import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
import matplotlib.patches as patches

# write your script here, we recommend the above libraries for making your animation
from LucasKanade import LucasKanade

frames = np.load('../data/carseq.npy')
#print(frames.shape) # (240, 320, 415)
# plt.imshow(frames[:,:,0], cmap='gray')
# plt.show()

rect_init = np.array([[59], [116], [145], [151]]).astype(np.float32)
rect = np.copy(rect_init)

width = rect[2] - rect[0]
height = rect[3] - rect[1]
#fig, ax = plt.subplots()
for i in range(0, frames.shape[2]-1):

    p = LucasKanade(frames[:,:,i], frames[:,:,i+1], rect)
    rect[0] += p[1]
    rect[1] += p[0]
    rect[2] = rect[0] + width
    rect[3] = rect[1] + height
    print(rect)
    print("frame ", i)
    pn = LucasKanade(frames)
    img = frames[:,:,i].copy()
    #cv2.rectangle(img,(rect[0],frames[:,:,i].shape[1]-rect[1]),(rect[2],frames[:,:,i].shape[1]-rect[3]), 240)
    fig, ax = plt.subplots()
    plt.imshow(img, cmap='gray')

    ax.add_patch(patches.Rectangle((rect[0], rect[1]), rect[2]-rect[0], rect[3]-rect[1], linewidth=2, edgecolor='green', fill=False))
    #plt.pause(0.1)
    #plt.show()
    #time.sleep(3)
    plt.pause(0.1)
    plt.close()
