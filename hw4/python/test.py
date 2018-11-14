from submission import *
from helper import *

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

data = np.load('../data/some_corresp.npz')
data_noisy = np.load('../data/some_corresp_noisy.npz')
intrinsics = np.load('../data/intrinsics.npz')

pts1 = data['pts1']
pts2 = data['pts2']
pts1_noisy = data_noisy['pts1']
pts2_noisy = data_noisy['pts2']

K1 = intrinsics['K1']
K2 = intrinsics['K2']

im1 = plt.imread('../data/im1.png')
im2 = plt.imread('../data/im2.png')

M = max(im1.shape)
num_points = pts1.shape[0]

''' Q2.1 '''
F = eightpoint(pts1, pts2, M)
np.savez('q2_1.npz', F=F, M=M)
# displayEpipolarF(im1, im2, F)

''' Q2.2 '''
indexes = np.random.choice(num_points, 7)

p1 = np.zeros((7, 2))
p2 = np.zeros((7, 2))
for i in range(indexes.shape[0]):
	p1[i] = pts1[indexes[i]]
	p2[i] = pts2[indexes[i]]

F7 = sevenpoint(p1, p2, M)
np.savez('q2_2.npz', F=F7, M=M, pts1=p1, pts2=p2)
# # print(len(F7))
# for f in F7:
# 	print(f)
# 	displayEpipolarF(im1, im2, f)

''' Q4.1 '''
F = eightpoint(pts1, pts2, M)
# epipolarMatchGUI(im1, im2, F)
np.savez('q4_1.npz', F=F, pts1=pts1, pts2=pts2)

''' Q5.1 '''
F_noisy = eightpoint(pts1_noisy, pts2_noisy, M)
# displayEpipolarF(im1, im2, F_noisy)
F, inliers = ransacF(pts1_noisy, pts2_noisy, M)
# displayEpipolarF(im1, im2, F)

''' Q5.3 '''
pts1_in = np.empty((0, 2))
pts2_in = np.empty((0, 2))

for i in range(inliers.shape[0]):
    if inliers[i] == True:
        pts1_in = np.append(pts1_in, pts1_noisy[i].reshape(1, 2), axis=0)
        pts2_in = np.append(pts2_in, pts2_noisy[i].reshape(1, 2), axis=0)
print("Num inliers: ", pts1_in.shape[0])
F = refineF(F, pts1_in, pts2_in)

E = essentialMatrix(F, K1, K2)
M1 = np.array([[1.0, 0, 0, 0],
               [0, 1.0, 0, 0],
               [0, 0, 1.0, 0]])
M2s = camera2(E)
C1 = np.dot(K1, M1)

min_error = 1e12
min_M2 = 0
min_C2 = 0
min_P = 0
min_index = 0

for i in range(4):
    C2 = np.dot(K2, M2s[:, :, i])
    P, error = triangulate(C1, pts1_in, C2, pts2_in)
    if error < min_error:
        if np.min(P[:, 2] >= 0):
            # print("Found!")
            min_error = error
            min_index = i
            min_M2 = M2s[:, :, min_index]
            min_C2 = C2
            min_P = P

M2 = np.copy(min_M2)
C2 = np.copy(min_C2)
P = np.copy(min_P)
print("Error before bundleAdjustment: ", min_error)

M2_ba, P_ba = bundleAdjustment(K1, M1, pts1_in, K2, M2, pts2_in, P)
C2_ba = np.dot(K2, M2_ba)
P_ba, error = triangulate(C1, pts1_in, C2_ba, pts2_in)
print("Error after bundleAdjustment: ", error)
# print(P.shape)
# print(P_ba.shape)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
xmin1, xmax1 = np.min(P_ba[:, 0]), np.max(P_ba[:, 0])
ymin1, ymax1 = np.min(P_ba[:, 1]), np.max(P_ba[:, 1])
zmin1, zmax1 = np.min(P_ba[:, 2]), np.max(P_ba[:, 2])
xmin2, xmax2 = np.min(P[:, 0]), np.max(P[:, 0])
ymin2, ymax2 = np.min(P[:, 1]), np.max(P[:, 1])
zmin2, zmax2 = np.min(P[:, 2]), np.max(P[:, 2])

xmin, xmax = min(xmin1, xmin2), max(xmax1, xmax2)
ymin, ymax = min(ymin1, ymin2), max(ymax1, ymax2)
zmin, zmax = min(zmin1, zmin2), max(zmax1, zmax2)
# xmin, xmax = -1, 1
# ymin, ymax = -0.5, 0.5
# zmin, zmax = 0, 2

ax.set_xlim3d(xmin, xmax)
ax.set_ylim3d(ymin, ymax)
ax.set_zlim3d(zmin, zmax)

ax.scatter(P_ba[:, 0], P_ba[:, 1], P_ba[:, 2], c='r', marker='o')
ax.scatter(P[:, 0], P[:, 1], P[:, 2], c='b', marker='o')
plt.show()
