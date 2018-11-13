import numpy as np
import cv2
import matplotlib.pyplot as plt

def LucasKanadeAffine(It, It1):
	# Input:
	#	It: template image
	#	It1: Current image
	# Output:
	#	M: the Affine warp matrix [2x3 numpy array]
	# put your implementation here
	M = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
	p = np.array([[0.0], [0.0], [0.0], [0.0], [0.0], [0.0]])

	width = It.shape[1]
	height = It.shape[0]

	xv, yv = np.meshgrid(np.arange(height), np.arange(width))
	xv = np.reshape(xv, (1, xv.shape[0]*xv.shape[1]))
	yv = np.reshape(yv, (1, yv.shape[0]*yv.shape[1]))

	Iy, Ix = np.gradient(It)

	mask = np.ones((height, width)).astype(np.float32)
	A = np.zeros((width*height, 6)).astype(np.float32)
	b = np.zeros((width*height)).astype(np.float32)

	threshold = 0.02
	dp_norm = threshold

	while dp_norm >= threshold:

		warped_img = cv2.warpAffine(It, M, (It.shape[1], It.shape[0]))
		warped_gradX = cv2.warpAffine(Ix, M, (It.shape[1], It.shape[0])).reshape((1, It.shape[1]*It.shape[0]))
		warped_gradY = cv2.warpAffine(Iy, M, (It.shape[1], It.shape[0])).reshape((1, It.shape[1]*It.shape[0]))
		warped_mask = cv2.warpAffine(mask, M, (It.shape[1], It.shape[0]))

		masked_img = warped_mask * It1

		# Construct A and b matrices
		b = (warped_img - masked_img).reshape((width*height, 1))

		A[:, 0] = warped_gradX[:] * xv[:]
		A[:, 1] = warped_gradX[:] * yv[:]
		A[:, 2] = warped_gradX[:]
		A[:, 3] = warped_gradY[:] * xv[:]
		A[:, 4] = warped_gradY[:] * yv[:]
		A[:, 5] = warped_gradY[:]

		dp, residuals, rank, s = np.linalg.lstsq(A, b)
		dp_norm = np.linalg.norm(dp)
		#print('dp_norm: ', dp_norm)
		p += dp
		M[0][0] = 1.0 + p[0]
		M[0][1] = p[1]
		M[0][2] = p[2]
		M[1][0] = p[3]
		M[1][1] = 1.0 + p[4]
		M[1][2] = p[5]
	#print(M)
	#print('p: ', p)
	return M
