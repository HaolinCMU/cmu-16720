import numpy as np

import skimage
import skimage.measure
import skimage.color
import skimage.restoration
import skimage.filters
import skimage.morphology
import skimage.segmentation

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# takes a color image
# returns a list of bounding boxes and black_and_white image
def findLetters(image):
    bboxes = []
    bw = None
    # insert processing in here
    # one idea estimate noise -> denoise -> greyscale -> threshold -> morphology -> label -> skip small boxes 
    # this can be 10 to 15 lines of code using skimage functions
    height = image.shape[0]
    width = image.shape[1]

    gray = skimage.color.rgb2gray(image)

    blurred = skimage.filters.gaussian(gray, sigma=1.0)

    threshold = skimage.filters.threshold_otsu(blurred)
    binary = gray < threshold

    opened = skimage.morphology.binary_opening(binary)
    # closed = skimage.morphology.binary_closing(opened)

    labels = skimage.measure.label(opened)
    image_label_overlay = skimage.color.label2rgb(labels, image=image)

    # fig, ax = plt.subplots()
    # ax.imshow(image_label_overlay)

    bbox_padding = 0
    regions = skimage.measure.regionprops(labels)

    total_area = 0
    for region in regions:
        total_area += region.area
    mean_area = total_area / len(regions)

    # print(mean_area)

    bboxes = []
    for region in regions:
        # take regions with large enough areas
        if region.area >= mean_area / 2:
            # draw rectangle around segmented letters
            min_row, min_col, max_row, max_col = region.bbox

            # Top-left
            y1 = max(0, min_row - bbox_padding)
            x1 = max(0, min_col - bbox_padding)
            # Bottom-right
            y2 = min(height, max_row + bbox_padding)
            x2 = min(width, max_col + bbox_padding)

            # rect = mpatches.Rectangle((x1, y1), x2 - x1, y2 - y1,
            #                         fill=False, edgecolor='red', linewidth=1)
            # ax.add_patch(rect)

            bboxes.append(np.array([y1, x1, y2, x2]))
            # print("region area: ", (y2 - y1) * (x2 - x1))

    # ax.set_axis_off()
    # plt.tight_layout()
    # plt.show()

    bw = 1.0 - opened
    # bw = skimage.color.gray2rgb(bw)

    # plt.imshow(bw, cmap='gray')

    # plt.show()

    return bboxes, bw