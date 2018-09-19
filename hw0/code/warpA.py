import numpy as np
from numpy.linalg import inv

def warp(im, A, output_shape):
    """ Warps (h,w) image im using affine (3,3) matrix A
    producing (output_shape[0], output_shape[1]) output image
    with warped = A*input, where warped spans 1...output_size.
    Uses nearest neighbor interpolation."""
    height = output_shape[0]
    width = output_shape[1]

    im_warped = np.zeros((height, width), 'float')

    A_inv = inv(A)

    # hh, ww = np.meshgrid(range(height), range(width), indexing='ij')
    # coordinate_grid = np.array([hh, ww, np.ones((height, width))])

    # coordinate_grid = np.matmul(A_inv, coordinate_grid)
    # coordinate_grid = np.rint(coordinate_grid)

    # print(coordinate_grid)

    for i in range(0, height):
      for j in range(0, width):
        coordinate_destination = np.array([i, j, 1]).transpose()
        
        coordinate_source = A_inv.dot(coordinate_destination)
        coordinate_source = np.rint(coordinate_source)

        x = int(coordinate_source[0])
        y = int(coordinate_source[1])

        if x < 0 or x >= height:
          im_warped[i][j] = 0
        elif y < 0 or y >= width:
          im_warped[i][j] = 0
        else:
          im_warped[i][j] = im[x][y]
      
    return im_warped
