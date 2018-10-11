import numpy as np
import cv2
from BRIEF import briefLite, briefMatch, plotMatches

def computeH(p1, p2):
    '''
    INPUTS:
        p1 and p2 - Each are size (2 x N) matrices of corresponding (x, y)'
                 coordinates between two images
    OUTPUTS:
     H2to1 - a 3 x 3 matrix encoding the homography that best matches the linear
            equation
    '''
    assert(p1.shape[1]==p2.shape[1])
    assert(p1.shape[0]==2)
    #############################
    # TO DO ...
    n = p1.shape[1]
    A = np.zeros((2*n, 9))

    for i in range(n):
        x = p1[0, i]
        y = p1[1, i]
        u = p2[0, i]
        v = p2[1, i]

        A[2*i] = [0, 0, 0, -u, -v, -1, y*u, y*v, y]
        A[2*i+1] = [u, v, 1, 0, 0, 0, -x*u, -x*v, -x]

    #print(A)
    (U, S, V) = np.linalg.svd(A)

    L = V[-1,:] / V[-1,-1]

    H2to1 = L.reshape(3,3)

    # (U, S, V) = np.linalg.svd(A, False)
    # H2to1 = np.reshape(V[:,8], (3,3))

    return H2to1

def ransacH(matches, locs1, locs2, num_iter=5000, tol=2):
    '''
    Returns the best homography by computing the best set of matches using
    RANSAC
    INPUTS
        locs1 and locs2 - matrices specifying point locations in each of the images
        matches - matrix specifying matches between these two sets of point locations
        nIter - number of iterations to run RANSAC
        tol - tolerance value for considering a point to be an inlier

    OUTPUTS
        bestH - homography matrix with the most inliers found during RANSAC
    '''
    ###########################
    # TO DO ...
    n = matches.shape[0]
    print("Number of matches: ", n)
    max_num_inliers = -1
    bestH = np.zeros((3,3))

    X = np.zeros((n, 2))
    U = np.zeros((n, 2))
    X[:, :] = locs1[matches[:, 0], 0:2]
    # X[:, [0,1]] = X[:, [1,0]]
    U[:, :] = locs2[matches[:, 1], 0:2]
    # U[:, [0,1]] = X[:, [1,0]]

    p1 = np.zeros((2, 4))
    p2 = np.zeros((2, 4))

    for iter in range(num_iter):
        indexs = np.random.choice(n, 4, replace=False)
        for i in range(indexs.shape[0]):
            x = locs1[matches[indexs[i], 0], 0:2]
            u = locs2[matches[indexs[i], 1], 0:2]
            p1[:, i] = x
            p2[:, i] = u

        H = computeH(p1, p2)
        # Compute number of inliers
        X_homo = np.append(np.transpose(X), np.ones((1, n)), axis=0) # 3xn
        U_homo = np.append(np.transpose(U), np.ones((1, n)), axis=0) # 3xn
        reprojection = np.matmul(H, U_homo)
        reprojection_norm = np.divide(reprojection, reprojection[2, :])

        error = X_homo - reprojection_norm
        #print(error)
        num_inliers = 0
        for i in range(n):
            squared_dist = error[0, i]**2 + error[1, i]**2
            #print(squared_dist)
            if squared_dist <= tol**2:
                num_inliers += 1
        #print(num_inliers)

        if num_inliers > max_num_inliers:
            bestH = H
            max_num_inliers = num_inliers
    print("RANSAC max number of inliers: ", max_num_inliers)

    return bestH



if __name__ == '__main__':
    im1 = cv2.imread('../data/model_chickenbroth.jpg')
    im2 = cv2.imread('../data/chickenbroth_01.jpg')

    locs1, desc1 = briefLite(im1)
    locs2, desc2 = briefLite(im2)

    matches = briefMatch(desc1, desc2)
    plotMatches(im1,im2,matches,locs1,locs2)

    # P = np.random.randint(100, size=(2,4))
    # P_homo = np.append(P, np.ones((1, 4)), axis=0)
    # H = np.random.randint(10, size=(3,3))
    # H = H / np.linalg.norm(H, 2)
    # Q = np.matmul(H, P_homo)
    # Q_norm = np.divide(Q, Q[2,:])
    # H_ = computeH(Q_norm[:2, :], P_homo[:2, :])
    # print(H - H_)

    ransacH(matches, locs1, locs2, num_iter=5000, tol=2)
