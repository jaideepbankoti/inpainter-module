"""
    References:
        Paper Referenced: A. Criminisi, P. Perez and K. Toyama, "Region filling and object removal by exemplar-based image inpainting," in IEEE Transactions on Image Processing, vol. 13, no. 9, pp. 1200-1212, Sept. 2004, doi: 10.1109/TIP.2004.833105.
        Link to Paper: https://ieeexplore.ieee.org/abstract/document/1323101
        Referenced the following repository: https://github.com/igorcmoura/inpaint-object-remover
        https://numpy.org/doc/stable/reference/generated/numpy.argwhere.html
        https://stackoverflow.com/questions/21210479/converting-from-rgb-to-lab-colorspace-any-insight-into-the-range-of-lab-val
        https://scikit-image.org/docs/dev/api/skimage.filters.html#skimage.filters.laplace
        image1.jpg and mask1.jpg: self made
        image2.jpg and mask2.jpg: from the resources folder of the above github repository
        image3.jpg and mask3.jpg: self clicked
        image4.jpg and mask4.jpg: from the resources folder of the above github repository
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt                         # images plotted using matplotlib
from skimage.filters import laplace                     # laplace used to find contour d_omega; install the scikit-image package for this module

test_img = cv2.imread("image4.jpg")                     # change string for image
mask = cv2.imread("mask4.jpg").astype(np.uint8)         # change string for mask
test_img = cv2.cvtColor(test_img, cv2.COLOR_BGR2RGB)    # converted from BGR to RGB as we will display in the end using plt.imshow()
r, c, _ = test_img.shape
mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
mask = (mask/np.max(mask)).astype(np.float32)               # mask values normalized in the range 0 to 1
test_img = (test_img/np.max(test_img)).astype(np.float32)   # since we will use them for confidence and other operations

# initializing attributes
img_copy = np.copy(test_img)                            # this image will be updated at each iteration
mask_copy = np.copy(mask)                               # this mask will be altered at each iteration
confidence = mask_copy.astype(float)                    # confidence term C(p)
data = np.zeros_like(img_copy, dtype=np.float32)        # data term D(p)
patch_len = 9                                           # Patch length set in accordance with the paper, can be changed


# calculating unit vector
def evalnormal():
    x = np.array([[.25, 0, -.25], [.5, 0, -.5], [.25, 0, -.25]])    # this kernal after convolving normalizes along the horizontal component
    y = np.array([[-.25, -.5, -.25], [0, 0, 0], [.25, .5, .25]])    # this kernal after convolving normalizes along the vertical component
    # convolving the above x and y kernels with the mask_copy to get normals
    x_norm = cv2.filter2D(mask_copy.astype(float), -1, x)
    y_norm = cv2.filter2D(mask_copy.astype(float), -1, y)
    normal = np.dstack((x_norm, y_norm))    # stack arrays depth wise
    r, c, _ = normal.shape
    norm = np.sqrt(y_norm**2 + x_norm**2).reshape(r, c, 1).repeat(2, axis=2)
    norm[norm == 0] = 1
    unit = normal / norm
    return unit

# calculating the gradient
def evalgradient():
    r, c, _ = img_copy.shape
    img_gray = cv2.cvtColor(img_copy, cv2.COLOR_RGB2GRAY).astype(float)
    img_gray[mask_copy == 1] = None     # set to none to indicate pixels are empty
    grad = np.nan_to_num(np.array(np.gradient(img_gray)))   # to generate valid result, used np.nan_to_num
    grad_val = np.sqrt(grad[0] ** 2 + grad[1] ** 2)
    grad_max = np.zeros([r, c, 2])
    front_pos = np.argwhere(front == 1)     # indicates he positions of the contour/front
    for pos in front_pos:
        patch = [[max(0, pos[0] - (patch_len - 1) // 2), min(pos[0] + (patch_len - 1) // 2, r - 1)],
                 [max(0, pos[1] - (patch_len - 1) // 2), min(pos[1] + (patch_len - 1) // 2, c - 1)]]    # this gets the four corner indices of patch
        patch_y_grad = grad[0][patch[0][0]:patch[0][1] + 1, patch[1][0]:patch[1][1] + 1]                # getting the value at the respective patch location
        patch_x_grad = grad[1][patch[0][0]:patch[0][1] + 1, patch[1][0]:patch[1][1] + 1]
        patch_grad_val = grad_val[patch[0][0]:patch[0][1] + 1, patch[1][0]:patch[1][1] + 1]
        patch_max_pos = np.unravel_index(patch_grad_val.argmax(), patch_grad_val.shape)                 # to convert flat indices to tuple
        grad_max[pos[0], pos[1], 0] = patch_y_grad[patch_max_pos]
        grad_max[pos[0], pos[1], 1] = patch_x_grad[patch_max_pos]
    return grad_max

# to find the source patch with minimum distance to the target pixel's patch
def findSrcPatch(pixel):
    r, c, _ = img_copy.shape
    target_patch = [[max(0, pixel[0] - (patch_len - 1) // 2), min(pixel[0] + (patch_len - 1) // 2, r - 1)],
                    [max(0, pixel[1] - (patch_len - 1) // 2), min(pixel[1] + (patch_len - 1) // 2, c - 1)]]         # this gets the four corner indices of target patch
    patch_r, patch_c = (1 + target_patch[0][1] - target_patch[0][0]), (1 + target_patch[1][1] - target_patch[1][0]) # to get number of rows and columns of the patch
    match, least_diff = None, float("inf")          # a value of infinity is set to compare later to find minimum value
    # converting img_copy to lab image, in accordance with the paper, as calculating distance on LAB images is more sensible
    lab_Img = cv2.cvtColor(img_copy, cv2.COLOR_RGB2LAB)
    for i in range(r - patch_r + 1):
        for j in range(c - patch_c + 1):
            src_patch = [[i, i + patch_r - 1], [j, j + patch_c - 1]]
            if np.sum(mask_copy[src_patch[0][0]:src_patch[0][1] + 1, src_patch[1][0]:src_patch[1][1] + 1]) != 0:
                continue
            patch_mask = 1 - mask_copy[target_patch[0][0]:target_patch[0][1] + 1,
                             target_patch[1][0]:target_patch[1][1] + 1]     # getting the value at the respective patch location, the (1-mask) done to invert mask
            color_mask = cv2.cvtColor(patch_mask, cv2.COLOR_GRAY2RGB)       # patch_mask converted to color_mask so that values of all channels can be filtered
            target_data = (lab_Img[target_patch[0][0]:target_patch[0][1] + 1,
                           target_patch[1][0]:target_patch[1][1] + 1]) * color_mask     # patch data in the lab_img found and multiplied by mask to filter the in region area
            source_data = (lab_Img[src_patch[0][0]:src_patch[0][1] + 1,
                           src_patch[1][0]:src_patch[1][1] + 1]) * color_mask           # patch data in the lab_img found and multiplied by mask to filter the in region area
            sq_diff = np.sum((target_data - source_data) ** 2)      # finding the squared distance
            euc_dist = np.sqrt((target_patch[0][0] - src_patch[0][0]) ** 2 + (
                        target_patch[1][0] - src_patch[1][0]) ** 2)  # eucledian distance can be further used as a tie breaker, but mainly squared difference of data value required
            diff = sq_diff + euc_dist
            if diff < least_diff or match is None:
                least_diff = diff
                match = src_patch
    return match


# the iteration loop starts here, summing elements of the mask to see if all parts of the mask are covered
while np.sum(mask_copy) > 0:
    # to find the fill front : step 1a of the algorithm
    # the laplacian on the mask will give us the edges of the mask, will be positive at white region(1) and negative at black region(0)
    # We want the black region which is inside the mask, so positive values are required!
    front = (laplace(mask_copy) > 0).astype('uint8')

    # updating confidence for priority update
    updated_conf = np.copy(confidence)
    front_pos = np.argwhere(front == 1)
    r, c, _ = img_copy.shape
    for pos in front_pos:
        patch = [[max(0, pos[0] - (patch_len - 1) // 2), min(pos[0] + (patch_len - 1) // 2, r - 1)],
                 [max(0, pos[1] - (patch_len - 1) // 2), min(pos[1] + (patch_len - 1) // 2, c - 1)]]    # to get the four corners of the patch
        area = (1 + patch[0][1] - patch[0][0]) * (1 + patch[1][1] - patch[1][0])                        # patch height and width multiplied to get patch area
        updated_conf[pos[0], pos[1]] = np.sum(
            np.sum(confidence[patch[0][0]:patch[0][1] + 1, patch[1][0]:patch[1][1] + 1])) / area        # formulae used in accordance to the paper
    confidence = updated_conf

    # updating data for priority update
    normal = evalnormal()
    gradient = evalgradient()
    norm_grad = normal * gradient       # formulae used in accordance to the paper
    data = np.sqrt(norm_grad[:, :, 0] ** 2 + norm_grad[:, :, 1] ** 2) + 0.0001      # added the 0.0001 to ensure data value is not 0

    # updating priority : step 1b
    priority = confidence * data * front        # formulae from the paper

    # finding target pixel, the one with highest priority: step 2a, using this we get target pixel as a tuple
    target_pixel = np.unravel_index(priority.argmax(), priority.shape)

    # finding the source patch: step 2b
    source_patch = findSrcPatch(target_pixel)

    # getting the target_patch
    target_patch = [
        [max(0, target_pixel[0] - (patch_len - 1) // 2), min(target_pixel[0] + (patch_len - 1) // 2, r - 1)],
        [max(0, target_pixel[1] - (patch_len - 1) // 2), min(target_pixel[1] + (patch_len - 1) // 2, c - 1)]]       # getting the four corners of the patch
    new_conf = confidence[target_pixel[0], target_pixel[1]]
    # getting positions of the target_patch to update the confidence value: step 3
    positions = np.argwhere(mask_copy[target_patch[0][0]:target_patch[0][1] + 1, target_patch[1][0]:target_patch[1][1] + 1] == 1) + [target_patch[0][0], target_patch[1][0]]
    for pos in positions:
        confidence[pos[0], pos[1]] = new_conf
    source_data = img_copy[source_patch[0][0]:source_patch[0][1] + 1, source_patch[1][0]:source_patch[1][1] + 1]
    target_data = img_copy[target_patch[0][0]:target_patch[0][1] + 1, target_patch[1][0]:target_patch[1][1] + 1]
    # the data_mask is used to update only those regions that needs filling
    data_mask = mask_copy[target_patch[0][0]:target_patch[0][1] + 1, target_patch[1][0]:target_patch[1][1] + 1]
    color_mask = cv2.cvtColor(data_mask, cv2.COLOR_GRAY2RGB)    # data mask converted to color_mask so that values of all channels can be filtered
    color_mask = color_mask/np.max(color_mask)

    # updated the target pixel value here with the source data, and used linear blending with both source data and target data
    updated_data = source_data * color_mask + target_data * (1 - color_mask)
    img_copy[target_patch[0][0]:target_patch[0][1] + 1, target_patch[1][0]:target_patch[1][1] + 1] = updated_data       # setting the target patch to updated data
    mask_copy[target_patch[0][0]:target_patch[0][1] + 1,target_patch[1][0]:target_patch[1][1] + 1] = 0  # decreasing the mask, otherwise while will run infinite


plt.figure(figsize=(10, 5))
plt.subplot(121)
plt.imshow(test_img)        # this is the original image
plt.subplot(122)
plt.imshow(img_copy)        # img_copy is the final generated image
plt.show()