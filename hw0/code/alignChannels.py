import numpy as np

def computeSSD(A, B):
  return np.sum((A-B)**2)

def findMinErrorIndex(A, B, max_shift):
  min_error = computeSSD(A, B)
  min_i = 0
  min_j = 0

  for i in range(-max_shift, max_shift):
    for j in range(-max_shift, max_shift):

      temp = np.roll(B, i, axis=0)
      temp = np.roll(temp, j, axis=1)

      error = computeSSD(A, temp)

      if error < min_error:
        min_error = error
        min_i = i
        min_j = j

  return [min_i, min_j]

def alignChannels(red, green, blue):
    """Given 3 images corresponding to different channels of a color image,
    compute the best aligned result with minimum abberations

    Args:
      red, green, blue - each is a HxW matrix corresponding to an HxW image
    
    Returns:
      rgb_output - HxWx3 color image output, aligned as desired"""

    rgb = np.zeros((red.shape[0], red.shape[1], 3), 'uint8')
    rgb[...,0] = red

    [i, j] = findMinErrorIndex(red, green, 30)

    green = np.roll(green, i, axis=0)
    green = np.roll(green, j, axis=1)
    rgb[...,1] = green

    [i, j] = findMinErrorIndex(red, blue, 30)

    blue = np.roll(blue, i, axis=0)
    blue = np.roll(blue, j, axis=1)
    rgb[...,2] = blue

    return rgb

