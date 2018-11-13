import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
import matplotlib.patches as patches

# write your script here, we recommend the above libraries for making your animation
from LucasKanade import LucasKanade
from LucasKanadeBasis import LucasKanadeBasis

frames = np.load('../data/sylvseq.npy')
bases = np.load('../data/sylvbases.npy')
rect = np.array([[101], [61], [155], [107]]).astype(np.float32)

num_frames = frames.shape[2]
width = rect[2] - rect[0]
height = rect[3] - rect[1]

rectList = np.empty((0, 4))
rectList = np.append(rectList, rect.reshape((1,4)), axis=0)

fig = plt.figure(1)
ax = fig.add_subplot(111)
ax.set_title("LucasKanade Tracking with Appearance Bias")

# Run LK tracking (with basis) in real-time
print("Running LK tracking (with basis) in real-time...")
for i in range(0, frames.shape[2]-1):
	print("frame ", i)
	p = LucasKanadeBasis(frames[:,:,i], frames[:,:,i+1], rect, bases)

	rect[0] += p[1]
	rect[1] += p[0]
	rect[2] = rect[0] + width
	rect[3] = rect[1] + height

	rectList = np.append(rectList, rect.reshape((1,4)), axis=0)

	img = frames[:,:,i].copy()

	ax.add_patch(patches.Rectangle((rect[0], rect[1]), rect[2]-rect[0], rect[3]-rect[1], linewidth=2, edgecolor='green', fill=False))
	plt.imshow(img, cmap='gray')
	plt.pause(0.05)
	ax.clear()
np.save('sylvseqrects.npy', rectList)


# Run vanilla LK tracking in real-time
rect = np.array([[101], [61], [155], [107]]).astype(np.float32)
rectList = np.empty((0, 4))
rectList = np.append(rectList, rect.reshape((1,4)), axis=0)

print("Running vanilla LK tracking in real-time...")
for i in range(0, frames.shape[2]-1):
	print("frame ", i)
	p = LucasKanade(frames[:,:,i], frames[:,:,i+1], rect)

	rect[0] += p[1]
	rect[1] += p[0]
	rect[2] = rect[0] + width
	rect[3] = rect[1] + height

	rectList = np.append(rectList, rect.reshape((1,4)), axis=0)

	img = frames[:,:,i].copy()

	ax.add_patch(patches.Rectangle((rect[0], rect[1]), rect[2]-rect[0], rect[3]-rect[1], linewidth=2, edgecolor='green', fill=False))
	plt.imshow(img, cmap='gray')
	plt.pause(0.01)
	ax.clear()
np.save('sylvseqrects-vanillaLK.npy', rectList)

# Playback frames with rectangles
rectList = np.load('sylvseqrects.npy')
rectListVanillaLK = np.load('sylvseqrects-vanillaLK.npy')

print("Playing back...")
for i in range(0, num_frames-1):
	print("frame: ", i)
	img = frames[:,:,i].copy()

	rect = rectList[i, :]
	rect_vanilla = rectListVanillaLK[i, :]

	ax.add_patch(patches.Rectangle((rect[0], rect[1]), rect[2]-rect[0]+1, rect[3]-rect[1]+1, linewidth=2, edgecolor='red', fill=False))
	ax.add_patch(patches.Rectangle((rect_vanilla[0], rect_vanilla[1]), rect_vanilla[2]-rect_vanilla[0]+1, rect_vanilla[3]-rect_vanilla[1]+1, linewidth=2, edgecolor='green', fill=False))
	plt.imshow(img, cmap='gray')
	if i in [1, 200, 300, 350, 400]:
		plt.savefig("sylv"+str(i)+".png")
	plt.pause(0.01)
	ax.clear()
