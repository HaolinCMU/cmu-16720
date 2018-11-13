import numpy as np
import cv2
import matplotlib.pyplot as plt

def InverseCompositionAffine(It, It1):
	# Input:
	#	It: template image
	#	It1: Current image

	# Output:
	#	M: the Affine warp matrix [2x3 numpy array]

	# put your implementation here
	M = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])

	width = It.shape[1]
	height = It.shape[0]

	xv, yv = np.meshgrid(np.arange(height), np.arange(width))
	xv = np.reshape(xv, (1, xv.shape[0]*xv.shape[1]))
	yv = np.reshape(yv, (1, yv.shape[0]*yv.shape[1]))

	Iy, Ix = np.gradient(It1)
	Ix = Ix.reshape(-1)
	Iy = Iy.reshape(-1)

	mask = np.ones((height, width)).astype(np.float32)
	A = np.zeros((width*height, 6)).astype(np.float32)
	b = np.zeros((width*height)).astype(np.float32)

	A[:, 0] = Ix[:] * xv[:]
	A[:, 1] = Ix[:] * yv[:]
	A[:, 2] = Ix[:]
	A[:, 3] = Iy[:] * xv[:]
	A[:, 4] = Iy[:] * yv[:]
	A[:, 5] = Iy[:]
	A1 = np.dot(np.linalg.inv(np.dot(A.transpose(), A)), A.transpose())

	threshold = 0.1
	dp_norm = threshold

	while dp_norm >= threshold:

		warped_mask = cv2.warpAffine(mask, M, (It.shape[1], It.shape[0]))
		warped_img = cv2.warpAffine(It, M, (It.shape[1], It.shape[0]))

		# Construct A and b matrices
		b = (It1*warped_mask - warped_img).reshape((width*height, 1))

		dp = np.dot(A1, b)

		dp_norm = np.linalg.norm(dp)
		#print('dp_norm: ', dp_norm)
		dM = np.array([[1.0 + dp[0], dp[1], dp[2]], [dp[3], 1.0 + dp[4], dp[5]], [0.0, 0.0, 1.0]]).astype(np.float32)
		M = np.dot(M, np.linalg.inv(dM))
		#print(M)
		#print('p: ', p)
	return M
