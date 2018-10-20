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

	p = p0
	threshold = 0.001

	It1_x = np.arange(0, It1.shape[0], 1)
	It1_y = np.arange(0, It1.shape[1], 1)
	interp_spline_It1 = RectBivariateSpline(It1_x, It1_y, It1)

	It_x = np.arange(0, It.shape[0], 1)
	It_y = np.arange(0, It.shape[1], 1)
	interp_spline_It = RectBivariateSpline(It_x, It_y, It)

	dp_norm = threshold

	x1 = rect[0]
	y1 = rect[1]
	x2 = rect[2]
	y2 = rect[3]

	width = int(x2 - x1 + 1)
	height = int(y2 - y1 + 1)

	while dp_norm >= threshold:
		A = np.zeros((int(width*height), 2))
		b = np.zeros((int(width*height)))
		indexes = np.zeros((int(width*height), 2))
		indexes_orig = np.zeros((int(width*height), 2))

		for i in range(width):
			for j in range(height):
				indexes[i*height+j, 0] = x1+i+p[0]+1
				indexes[i*height+j, 1] = y1+j+p[1]+1
				indexes_orig[i*height+j, 0] = x1+i+1
				indexes_orig[i*height+j, 1] = y1+j+1

		I = interp_spline_It1.ev(indexes[:,0], indexes[:,1])
		Ix = interp_spline_It1.ev(indexes[:,0], indexes[:,1], dx=1)
		Iy = interp_spline_It1.ev(indexes[:,0], indexes[:,1], dy=1)
		I_orig = interp_spline_It.ev(indexes_orig[:,0], indexes_orig[:,1])

		# Construct A and b matrices
		b[:] = (I_orig - I)[:]
		A[:, 0] = Ix[:]
		A[:, 1] = Iy[:]

		dp, residuals, rank, s = np.linalg.lstsq(A, b)

		dp_norm = np.linalg.norm(dp)
		print('dp_norm: ', dp_norm)
		p += dp
		print('p: ', p)

	return p
