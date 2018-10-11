import numpy as np
import imageio
from planarH import computeH
import matplotlib.pyplot as plt

def compute_extrinsics(K, H):

    H0 = np.matmul(np.linalg.inv(K), H)
    x = H0[:, 0:2]
    U, S, V = np.linalg.svd(x, full_matrices=True)

    y = np.array([[1, 0],
                  [0, 1],
                  [0, 0]])

    R = -np.matmul(np.matmul(U, y), V)
    R3 = np.cross(R[:, 0], R[:, 1]).reshape(3, 1)
    R = np.append(R, R3, axis=1)
    if np.linalg.det(R) == -1:
        R[:, 2] *= -1

    sum = 0
    for i in range(3):
        for j in range(2):
            sum = sum + H0[i,j] / R[i,j]
    lamda = sum / 6

    t = (H0[:, 2] / lamda).reshape(3, 1)

    return R, t


def project_extrinsics(K, W, R, t):

    x = np.matmul(np.matmul(K, np.append(R, t, axis=1)), W)
    X = (x[0:2, :] / x[2, :]).astype(int)

    return X


if __name__ == '__main__':

    W = np.array([[0.0, 18.2, 18.2, 0.0 ],
                  [0.0, 0.0,  26.0, 26.0],
                  [0.0, 0.0,  0.0,  0.0 ]])

    X = np.array([[483, 1704, 2175, 67  ],
                  [810, 781,  2217, 2286]])

    K = np.array([[3043.72, 0.0,      1196.00 ],
                  [0.0,     3043.72,  1604.00],
                  [0.0,     0.0,      1.0]])

    with open('../data/sphere.txt', "r") as f:
        str = f.read()
    lines = str.split('\n')
    x = lines[0].split('  ')
    y = lines[1].split('  ')
    z = lines[2].split('  ')

    sphere = []
    for i in range(1, len(x)):
        sphere.append([float(x[i]), float(y[i]), float(z[i])])
    W_sphere = np.array(sphere).transpose()
    W_sphere = np.append(W_sphere, np.ones((1, W_sphere.shape[1])), axis=0)

    W_xy = W[0:2, :]
    H = computeH(X, W_xy)
    R, t = compute_extrinsics(K, H)

    dots = project_extrinsics(K, W_sphere, R, t) + np.array((350, 820)).reshape((2,1))

    im = imageio.imread('../data/prince_book.jpeg')
    fig = plt.figure()
    plt.imshow(im)
    # plt.show()
    for i in range(dots.shape[1]):
        plt.plot(dots[0, i], dots[1, i], 'y.', markersize=1)
    plt.draw()
    plt.waitforbuttonpress(0)
    plt.close(fig)
