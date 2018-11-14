'''
Q3.3:
    1. Load point correspondences
    2. Obtain the correct M2
    3. Save the correct M2, C2, and P to q3_3.npz
'''
import numpy as np
import matplotlib as plt
from submission import *
from helper import *

data = np.load('../data/some_corresp.npz')
intrinsics = np.load('../data/intrinsics.npz')

pts1 = data['pts1']
pts2 = data['pts2']
K1 = intrinsics['K1']
K2 = intrinsics['K2']


im1 = plt.imread('../data/im1.png')
im2 = plt.imread('../data/im2.png')

M = max(im1.shape[0], im1.shape[1])

F = eightpoint(pts1, pts2, M)
E = essentialMatrix(F, K1, K2)

print('F: ')
for rows in F:
    print(rows)

print('E: ')
for rows in E:
    print(rows)

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
    P, error = triangulate(C1, pts1, C2, pts2)
    if error < min_error:
        min_error = np.copy(error)
        min_index = np.copy(i)
        min_C2 = np.copy(C2)
        min_P = np.copy(P)
np.savez('q3_3.npz', M2=M2s[:,:,min_index], C2=min_C2, P=min_P)
print("find M2 error: ", min_error)
