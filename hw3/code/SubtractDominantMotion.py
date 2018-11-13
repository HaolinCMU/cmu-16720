import numpy as np
import cv2
from scipy.ndimage.morphology import binary_dilation, binary_erosion, binary_opening, binary_closing

from LucasKanadeAffine import LucasKanadeAffine
from InverseCompositionAffine import InverseCompositionAffine

def SubtractDominantMotion(image1, image2):
	# Input:
	#	Images at time t and t+1
	# Output:
	#	mask: [nxm]
    # put your implementation here

    mask = np.ones(image1.shape, dtype=bool)

    ROI = np.zeros(image1.shape, dtype=np.float32)
    ROI[10:150, 60:220] = 1.0

    threshold = 0.21

    M = LucasKanadeAffine(image1, image2)
    # M = InverseCompositionAffine(image1, image2)

    warped_img = cv2.warpAffine(image1, M, (image1.shape[1], image1.shape[0]))
    diff = np.absolute(warped_img - image1) * ROI
    mask = diff > threshold
    # mask = binary_erosion(mask, np.ones((2, 2)))
    # mask = binary_dilation(mask, np.ones((2, 2)))

    mask = binary_dilation(mask, np.ones((10, 10)))
    # mask = binary_erosion(mask, np.ones((7, 7)))
    # mask = binary_erosion(mask, np.ones((2, 10)))
    # mask = binary_opening(mask, np.ones((4,4)))
    # mask = binary_closing(mask, np.ones((4,4)))


    return mask
