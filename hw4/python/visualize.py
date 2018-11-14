'''
Q4.2:
    1. Integrating everything together.
    2. Loads necessary files from ../data/ and visualizes 3D reconstruction using scatter
'''
from submission import *
from helper import *

import numpy as np
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D

templeCoords = np.load('../data/templeCoords.npz')
data = np.load('../data/some_corresp.npz')
intrinsics = np.load('../data/intrinsics.npz')

pts1 = data['pts1']
pts2 = data['pts2']

x1 = templeCoords['x1']
y1 = templeCoords['y1']
num_points = x1.shape[0]

x2 = np.zeros((num_points, 1))
y2 = np.zeros((num_points, 1))

K1 = intrinsics['K1']
K2 = intrinsics['K2']

im1 = plt.imread('../data/im1.png')
im2 = plt.imread('../data/im2.png')

M = max(im1.shape)
num_points = x1.shape[0]

F = eightpoint(pts1, pts2, M)
E = essentialMatrix(F, K1, K2)

for i in range(num_points):
    x2[i], y2[i] = epipolarCorrespondence(im1, im2, F, x1[i], y1[i])
points1 = np.append(x1, y1, axis=1)
points2 = np.append(x2, y2, axis=1)

M1 = np.array([[1.0, 0, 0, 0],
               [0, 1.0, 0, 0],
               [0, 0, 1.0, 0]])
M2s = camera2(E)

C1 = np.dot(K1, M1)

min_error = 1e12
min_C2 = 0
min_P = 0
min_index = 0

for i in range(4):
    C2 = np.dot(K2, M2s[:, :, i])
    P, error = triangulate(C1, points1, C2, points2)
    if error < min_error:
        if np.min(P[:, 2] >= 0):
            print("Found!")
            min_error = np.copy(error)
            min_index = np.copy(i)
            min_C2 = np.copy(C2)
            min_P = np.copy(P)
np.savez('q4_2.npz', M1=M1, M2=M2s[:,:,min_index], C1=C1, C2=min_C2)
print("find M2 error: ", min_error)
P = np.copy(min_P)
# print(P)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
xmin, xmax = np.min(P[:, 0]), np.max(P[:, 0])
ymin, ymax = np.min(P[:, 1]), np.max(P[:, 1])
zmin, zmax = np.min(P[:, 2]), np.max(P[:, 2])
# xmin, xmax = -1, 1
# ymin, ymax = -0.5, 0.5
# zmin, zmax = 0, 2

ax.set_xlim3d(xmin, xmax)
ax.set_ylim3d(ymin, ymax)
ax.set_zlim3d(zmin, zmax)

ax.scatter(P[:, 0], P[:, 1], P[:, 2], c='b', marker='o')
plt.show()
