import numpy as np
from scipy.interpolate import RectBivariateSpline

def LucasKanade(It, It1, rect, p0 = np.zeros(2)):
	# Input:
	#	It: template image
	#	It1: Current image
	#	rect: Current position of the car: [x1, y1, x2, y2]^T
	#	(top left, bot right coordinates)
	#	p0: Initial movement vector [dp_x0, dp_y0]
	# Output:
	#	p: movement vector [dp_x, dp_y]

	# Put your implementation here

	p = np.copy(p0)
	threshold = 0.01

	It1_x = np.arange(0, It1.shape[0], 1)
	It1_y = np.arange(0, It1.shape[1], 1)
	interp_spline_It1 = RectBivariateSpline(It1_x, It1_y, It1)

	It_x = np.arange(0, It.shape[0], 1)
	It_y = np.arange(0, It.shape[1], 1)
	interp_spline_It = RectBivariateSpline(It_x, It_y, It)

	y1 = rect[0]
	x1 = rect[1]
	y2 = rect[2]
	x2 = rect[3]

	width = int(np.rint(x2 - x1 + 1))
	height = int(np.rint(y2 - y1 + 1))

	yv, xv = np.meshgrid(np.arange(height), np.arange(width))
	xv = xv.astype(np.float32)
	yv = yv.astype(np.float32)
	xv += x1
	yv += y1
	xv = np.reshape(xv, (1, xv.shape[0]*xv.shape[1]))
	yv = np.reshape(yv, (1, yv.shape[0]*yv.shape[1]))
	xv_orig = np.copy(xv)
	yv_orig = np.copy(yv)

	A = np.zeros((int(width*height), 2))
	b = np.zeros((int(width*height)))

	I_orig = interp_spline_It.ev(xv_orig, yv_orig)
	# xv = xv_orig + p0[0]
	# yv = yv_orig + p0[1]
	indexes = np.zeros((int(width*height), 2))
	indexes_orig = np.zeros((int(width*height), 2))

	dp_norm = threshold
	while dp_norm >= threshold:

		xv = xv_orig + p[0]
		yv = yv_orig + p[1]

		I = interp_spline_It1.ev(xv, yv)
		Ix = interp_spline_It1.ev(xv, yv, dx=1)
		Iy = interp_spline_It1.ev(xv, yv, dy=1)

		# Construct A and b matrices
		b[:] = (I_orig - I)[:]
		A[:, 0] = Ix[:]
		A[:, 1] = Iy[:]

		dp, residuals, rank, s = np.linalg.lstsq(A, b)

		dp_norm = np.linalg.norm(dp)
		#print('dp_norm: ', dp_norm)
		p += dp
		#print('p: ', p)

	return p
