"""
Homework4.
Replace 'pass' by your implementation.
"""

# Insert your package here
import numpy as np
import numpy.matlib as matlib
import matplotlib.pyplot as plt
import sympy as sp
import scipy
from scipy.ndimage.filters import gaussian_filter
from math import *
from helper import *

'''
Q2.1: Eight Point Algorithm
	Input:  pts1, Nx2 Matrix
	pts2, Nx2 Matrix
	M, a scalar parameter computed as max (imwidth, imheight)
	Output: F, the fundamental matrix
'''
def eightpoint(pts1, pts2, M):

	num_points = pts1.shape[0]

	pts1 = np.divide(pts1, np.matlib.repmat(M, num_points, pts1.shape[1]))
	pts2 = np.divide(pts2, np.matlib.repmat(M, num_points, pts2.shape[1]))

	A = np.zeros((num_points, 9))
	A[:, 0] = pts2[:, 0] * pts1[:, 0]
	A[:, 1] = pts2[:, 0] * pts1[:, 1]
	A[:, 2] = pts2[:, 0]
	A[:, 3] = pts2[:, 1] * pts1[:, 0]
	A[:, 4] = pts2[:, 1] * pts1[:, 1]
	A[:, 5] = pts2[:, 1]
	A[:, 6] = pts1[:, 0]
	A[:, 7] = pts1[:, 1]
	A[:, 8] = np.ones(num_points)

	U, S, V = np.linalg.svd(A)
	f = V.transpose()[:, -1]

	F = f.reshape((3, 3))

	UF, SF, VF = np.linalg.svd(F)
	d1 = SF[0]
	d2 = SF[1]
	d3 = 0
	S = np.diag([d1, d2, d3])
	F = np.dot(np.dot(UF, S), VF)

	F = refineF(F, pts1, pts2)
	# Un-normalize
	T = np.diag([1.0/M, 1.0/M, 1.0])

	F = np.dot(np.dot(T.transpose(), F), T)

	# im1 = plt.imread('../data/im1.png')
	# im2 = plt.imread('../data/im2.png')
	# displayEpipolarF(im1, im2, F)
	# print(F)
	#np.savez('q2_1.npz', F=F, M=M)

	return F


'''
Q2.2: Seven Point Algorithm
	Input:  pts1, Nx2 Matrix
	pts2, Nx2 Matrix
	M, a scalar parameter computed as max (imwidth, imheight)
	Output: Farray, a list of estimated fundamental matrix.
'''
def sevenpoint(pts1, pts2, M):

	num_points = pts1.shape[0]

	pts1 = np.divide(pts1, np.matlib.repmat(M, num_points, pts1.shape[1]))
	pts2 = np.divide(pts2, np.matlib.repmat(M, num_points, pts2.shape[1]))

	A = np.zeros((num_points, 9))
	A[:, 0] = pts2[:, 0] * pts1[:, 0]
	A[:, 1] = pts2[:, 0] * pts1[:, 1]
	A[:, 2] = pts2[:, 0]
	A[:, 3] = pts2[:, 1] * pts1[:, 0]
	A[:, 4] = pts2[:, 1] * pts1[:, 1]
	A[:, 5] = pts2[:, 1]
	A[:, 6] = pts1[:, 0]
	A[:, 7] = pts1[:, 1]
	A[:, 8] = np.ones(num_points)

	U, S, V = np.linalg.svd(A)
	F1 = V.transpose()[:, -1].reshape((3, 3))
	F2 = V.transpose()[:, -2].reshape((3, 3))

	fun = lambda a: np.linalg.det(a * F1 + (1 - a) * F2)

	a0 = fun(0)
	a1 = 2.0 * (fun(1) - fun(-1)) / 3 - (fun(2) - fun(-2)) / 12
	a2 = 0.5 * fun(1) + 0.5 * fun(-1) - fun(0)
	#a3 = -1.0/6 * fun(1) + 1.0/6 * fun(-1) + 1.0/12 * fun(2) - 1.0/12 * fun(-2)
	a3 = fun(1) - a0 - a1 - a2
	roots = np.roots(np.array([a3, a2, a1, a0]))

	T = np.diag([1.0/M, 1.0/M, 1.0])

	Farray = []
	for alpha in roots:

		F = F1 * float(np.real(alpha)) + F2 * (1 - float(np.real(alpha)))
		U, S, V = np.linalg.svd(F)

		ss = np.diag([S[0], S[1], S[2]])
		F = np.dot(np.dot(U, ss), V)
		F = np.dot(np.dot(T.transpose(), F), T)
		Farray.append(F)

	# print('Num of F: ', len(Farray))

	# im1 = plt.imread('../data/im1.png')
	# im2 = plt.imread('../data/im2.png')
	# for f in Farray:
	# 	displayEpipolarF(im1, im2, f)

	return Farray

'''
Q3.1: Compute the essential matrix E.
	Input:  F, fundamental matrix
	K1, internal camera calibration matrix of camera 1
	K2, internal camera calibration matrix of camera 2
	Output: E, the essential matrix
'''
def essentialMatrix(F, K1, K2):
	return np.dot(np.dot(K2.transpose(), F), K1)


'''
Q3.2: Triangulate a set of 2D coordinates in the image to a set of 3D points.
	Input:  C1, the 3x4 camera matrix
	pts1, the Nx2 matrix with the 2D image coordinates per row
	C2, the 3x4 camera matrix
	pts2, the Nx2 matrix with the 2D image coordinates per row
	Output: P, the Nx3 matrix with the corresponding 3D points per row
	err, the reprojection error.
'''
def triangulate(C1, pts1, C2, pts2):
	num_points = pts1.shape[0]

	P = np.zeros((num_points, 3))
	A = np.zeros((4, 4))
	for i in range(num_points):
		A[0, :] = pts1[i, 0] * C1[2, :] - C1[0, :]
		A[1, :] = pts1[i, 1] * C1[2, :] - C1[1, :]
		A[2, :] = pts2[i, 0] * C2[2, :] - C2[0, :]
		A[3, :] = pts2[i, 1] * C2[2, :] - C2[1, :]

		U, S, V = np.linalg.svd(A)
		p = V.transpose()[:, -1]

		P[i, :] = np.divide(p[0:3], p[3])
	# print(P)

	# P_homo = np.copy(P)
	# P_homo[:, 0] = np.divide(P[:, 0], P[:, 2])
	# P_homo[:, 1] = np.divide(P[:, 1], P[:, 2])
	# P_homo[:, 2] = np.ones((P.shape[0]))

	P_homo = np.append(P, np.ones((P.shape[0], 1)), axis=1).transpose()


	# print(P_homo)
	#P = np.append(P, np.ones(num_points), axis=1)
	# p1_reprojected = np.dot(C1, P)
	# p2_reprojected = np.dot(C2, P)
	p1_reprojected = np.dot(C1, P_homo)
	p2_reprojected = np.dot(C2, P_homo)
	# print(p1_reprojected)

	p1_normalized = np.zeros((2, num_points))
	p2_normalized = np.zeros((2, num_points))

	p1_normalized[0, :] = np.divide(p1_reprojected[0, :], p1_reprojected[2, :])
	p1_normalized[1, :] = np.divide(p1_reprojected[1, :], p1_reprojected[2, :])
	p2_normalized[0, :] = np.divide(p2_reprojected[0, :], p2_reprojected[2, :])
	p2_normalized[1, :] = np.divide(p2_reprojected[1, :], p2_reprojected[2, :])
	p1_normalized = p1_normalized.transpose()
	p2_normalized = p2_normalized.transpose()

	# print(p1_normalized - pts1)

	p1_error = (p1_normalized - pts1)[:, 0] * (p1_normalized - pts1)[:, 0] + (p1_normalized - pts1)[:, 1] * (p1_normalized - pts1)[:, 1]
	p2_error = (p2_normalized - pts2)[:, 0] * (p2_normalized - pts2)[:, 0] + (p2_normalized - pts2)[:, 1] * (p2_normalized - pts2)[:, 1]
	error = np.sum(p1_error) + np.sum(p2_error)
	# print(error)
	return P, error


'''
Q4.1: 3D visualization of the temple images.
	Input:  im1, the first image
	im2, the second image
	F, the fundamental matrix
	x1, x-coordinates of a pixel on im1
	y1, y-coordinates of a pixel on im1
	Output: x2, x-coordinates of the pixel on im2
	y2, y-coordinates of the pixel on im2

'''
def epipolarCorrespondence(im1, im2, F, x1, y1):
	window = 10
	im1 = np.copy(im1.astype(float))
	im2 = np.copy(im2.astype(float))

	point = np.array([[x1], [y1], [1]])
	l = np.dot(F, point)
	l = l / np.linalg.norm(l)

	pts_xy = np.empty((0, 2))
	pts_yx = np.empty((0, 2))

	if l[0] != 0:
		for y in range(window, im2.shape[0] - window):
			x = floor(-1.0 * (l[1] * y + l[2]) / l[0])
			if x >= window and x <= im2.shape[1] - window:
				pts_yx = np.append(pts_yx, np.array([x, y]).reshape(1, 2), axis=0)


	else:
		for x in range(window, im2.shape[1] - window):
			y = floor(-1.0 * (l[0] * x + l[2]) / l[1])
			if y >= window and y <= im2.shape[0] - window:
				pts_xy = np.append(pts_xy, np.array([x, y]).reshape(1, 2), axis=0)

	pts = pts_yx

	patch1 = im1[int(y1 - window + 1) : int(y1 + window), int(x1 - window + 1) : int(x1 + window), :]

	min_error = 1e12
	min_index = 0

	for i in range(pts.shape[0]):
		x2 = pts[i, 0]
		y2 = pts[i, 1]
		if sqrt((x1-x2)**2 + (y1-y2)**2) < 50:
			patch2 = im2[int(y2 - window + 1) : int(y2 + window), int(x2 - window + 1) : int(x2 + window), :]
			error = patch1 - patch2
			error_filtered = np.sum(gaussian_filter(error, sigma=1.0))
			if error_filtered < min_error:
				min_error = error_filtered
				min_index = i
	x2 = pts[min_index, 0]
	y2 = pts[min_index, 1]

	return x2, y2


'''
Q5.1: RANSAC method.
	Input:  pts1, Nx2 Matrix
	pts2, Nx2 Matrix
	M, a scaler parameter
	Output: F, the fundamental matrix
	inliers, Nx1 bool vector set to true for inliers
'''
def ransacF(pts1, pts2, M):
	num_iter = 500
	threshold = 0.001

	num_points = pts1.shape[0]
	pts1 = np.append(pts1, np.ones((num_points, 1)), axis=1)
	pts2 = np.append(pts2, np.ones((num_points, 1)), axis=1)

	max_inliers = 0
	inliers = np.zeros((num_points, 1), dtype=bool)
	inliers_index_final = []

	points1 = np.zeros((7, 2))
	points2 = np.zeros((7, 2))

	final_points1 = []
	final_points2 = []
	F = []

	for i in range(num_iter):
		print("RANSAC iteration ", i)

		indexes = np.random.choice(num_points, 7)
		for id in range(indexes.shape[0]):
			points1[id] = pts1[indexes[id], 0:2]
			points2[id] = pts2[indexes[id], 0:2]

		inliers_index = []
		Farray = sevenpoint(points1, points2, M)
		for f in Farray:
			num_inliers = 0
			selected_points1 = np.empty((0, 2))
			selected_points2 = np.empty((0, 2))
			for k in range(num_points):
				p1 = np.array([pts1[k, 0], pts1[k, 1], 1.0])
				p2 = np.array([pts2[k, 0], pts2[k, 1], 1.0])
				error = abs(np.dot(np.dot(p2.transpose(), f), p1))

				if error < threshold:
					num_inliers += 1
					selected_points1 = np.append(selected_points1, pts1[k, 0:2].reshape(1, 2), axis=0)
					selected_points2 = np.append(selected_points2, pts2[k, 0:2].reshape(1, 2), axis=0)
					inliers_index.append(k)
					# inliers[k] = 1

			if num_inliers > max_inliers:
				final_points1 = np.copy(selected_points1)
				final_points2 = np.copy(selected_points2)
				inliers_index_final = np.array(inliers_index)
				max_inliers = num_inliers
				print("Max num inliers: ", max_inliers)
				print("inliers_index_final shape: ", inliers_index_final.shape)
				F = np.copy(f)

	for inlier_index in inliers_index_final:
		inliers[inlier_index] = 1

	F = eightpoint(final_points1, final_points2, M)

	return F, inliers



'''
Q5.2: Rodrigues formula.
	Input:  r, a 3x1 vector
	Output: R, a rotation matrix
'''
def rodrigues(r):
	theta = np.linalg.norm(r)
	R = np.zeros((3,3))
	if theta == 0:
		R = np.eye(3)
	else:
		axis = r / theta
		axis_sm = np.array([[0, -axis[2], axis[1]],
							[axis[2], 0, -axis[0]],
							[-axis[1], axis[0], 0]])
		R = cos(theta) * np.eye(3) + np.dot((1 - cos(theta)) * axis, axis.transpose()) + sin(theta) * axis_sm

	return R
'''
Q5.2: Inverse Rodrigues formula.
	Input:  R, a rotation matrix
	Output: r, a 3x1 vector
'''
def invRodrigues(R):
	epsilon = 1e-16
	theta = acos((np.trace(R) - 1) / 2.0)
	r = np.zeros((3, 1))

	if abs(theta) > epsilon:
		norm_axis = 1.0 / (2*sin(theta)) * np.array([[R[2, 1] - R[1, 2]],
													 [R[0, 2] - R[2, 0]],
													 [R[1, 0] - R[0, 1]]])
		r = theta * norm_axis
	return r


'''
Q5.3: Rodrigues residual.
	Input:  K1, the intrinsics of camera 1
	M1, the extrinsics of camera 1
	p1, the 2D coordinates of points in image 1
	K2, the intrinsics of camera 2
	p2, the 2D coordinates of points in image 2
	x, the flattened concatenationg of P, r2, and t2.
	Output: residuals, 4N x 1 vector, the difference between original and estimated projections
'''
def rodriguesResidual(K1, M1, p1, K2, p2, x):

	# rod_v = x[0:3].reshape(3, 1)
	# t = x[3:6].reshape(3, 1)
	# P = x[6:].reshape(-1, 3)
	rod_v = x[-6:-3].reshape(3, 1)
	t = x[-3:].reshape(3, 1)
	P = x[0:-6].reshape(-1, 3)


	R = rodrigues(rod_v);
	M2 = np.append(R, t, axis=1)
	C1 = np.dot(K1, M1)
	C2 = np.dot(K2, M2)

	P_homo = np.append(P, np.ones((P.shape[0], 1)), axis=1).transpose()

	p1_reprojected = np.dot(C1, P_homo)
	p2_reprojected = np.dot(C2, P_homo)

	p1_normalized = np.zeros((2, P_homo.shape[1]))
	p2_normalized = np.zeros((2, P_homo.shape[1]))

	p1_normalized[0, :] = np.divide(p1_reprojected[0, :], p1_reprojected[2, :])
	p1_normalized[1, :] = np.divide(p1_reprojected[1, :], p1_reprojected[2, :])
	p2_normalized[0, :] = np.divide(p2_reprojected[0, :], p2_reprojected[2, :])
	p2_normalized[1, :] = np.divide(p2_reprojected[1, :], p2_reprojected[2, :])
	p1_normalized = p1_normalized.transpose()
	p2_normalized = p2_normalized.transpose()

	error1 = (p1 - p1_normalized).reshape(-1)
	error2 = (p2 - p2_normalized).reshape(-1)

	residuals = np.append(error1, error2, axis=0)

	return residuals

'''
Q5.3 Bundle adjustment.
	Input:  K1, the intrinsics of camera 1
	M1, the extrinsics of camera 1
	p1, the 2D coordinates of points in image 1
	K2,  the intrinsics of camera 2
	M2_init, the initial extrinsics of camera 1
	p2, the 2D coordinates of points in image 2
	P_init, the initial 3D coordinates of points
	Output: M2, the optimized extrinsics of camera 1
	P2, the optimized 3D coordinates of points
'''
def bundleAdjustment(K1, M1, p1, K2, M2_init, p2, P_init):

	residual = lambda x: rodriguesResidual(K1, M1, p1, K2, p2, x)
	# residual = lambda x: (rodriguesResidual(K1, M1, p1, K2, p2, x)**2).sum()

	R2_init = M2_init[:, 0:3]
	t2_init = M2_init[:, 3]
	r2_init = invRodrigues(R2_init).reshape(-1)

	x_init = np.zeros(3 * P_init.shape[0] + 6)
	# x_init[0:3] = r2_init
	# x_init[3:6] = t2_init
	# x_init[6:] = P_init.reshape(-1)
	x_init[-6:-3] = r2_init
	x_init[-3:] = t2_init
	x_init[0:-6] = P_init.reshape(-1)



	x_optimized, _ = scipy.optimize.leastsq(residual, x_init)
	# x_optimized = scipy.optimize.minimize(residual, x_init)

	# r = x_optimized[0:3].reshape(3, 1)
	# t = x_optimized[3:6]
	# P = x_optimized[6:].reshape(-1, 3)
	r = x_optimized[-6:-3].reshape(3, 1)
	t = x_optimized[-3:]
	P = x_optimized[0:-6].reshape(-1, 3)

	R = rodrigues(r)
	M2 = np.append(R, t.reshape(3, 1), axis=1)

	return M2, P
