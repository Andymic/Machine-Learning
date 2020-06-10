# Author: Andy M.
# Last Modified: 6/9/20

import numpy as np
import matplotlib.pyplot as plt
from scipy import misc

i = misc.ascent()

plt.title('Original Image')
plt.grid(False)
plt.gray()
plt.axis('off')
plt.imshow(i)
plt.show()

i_transformed = np.copy(i)
size_x = i_transformed.shape[0]
size_y = i_transformed.shape[1]

print(i_transformed.shape)

# https://en.wikipedia.org/wiki/Kernel_(image_processing)
# This filter detects edges nicely
# It creates a convolution that only passes through sharp edges and straight
# lines.

# Experiment with different values for fun effects.
filter1 = [[0, 1, 0], [1, -4, 1], [0, 1, 0]]
filter2 = [[-1, -2, -1], [0, 0, 0], [1, 2, 1]]
filter3 = [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]

filters = []
filters.append(filter1)
filters.append(filter2)
filters.append(filter3)
# If all the digits in the filter don't add up to 0 or 1, you
# should probably do a weight to get it to do so
# so, for example, if your weights are 1,1,1 1,2,1 1,1,1
# They add up to 10, so you would set a weight of .1 if you want to normalize them
weight = 1
it = 0
i_transformed_ar = []
for filter in filters:
    for x in range(1, size_x - 1):
        for y in range(1, size_y - 1):
            convolution = 0.0
            convolution = convolution + (i[x - 1, y - 1] * filter[0][0])
            convolution = convolution + (i[x, y - 1] * filter[0][1])
            convolution = convolution + (i[x + 1, y - 1] * filter[0][2])
            convolution = convolution + (i[x - 1, y] * filter[1][0])
            convolution = convolution + (i[x, y] * filter[1][1])
            convolution = convolution + (i[x + 1, y] * filter[1][2])
            convolution = convolution + (i[x - 1, y + 1] * filter[2][0])
            convolution = convolution + (i[x, y + 1] * filter[2][1])
            convolution = convolution + (i[x + 1, y + 1] * filter[2][2])
            convolution = convolution * weight
            if (convolution < 0):
                convolution = 0
            if (convolution > 255):
                convolution = 255
            i_transformed[x, y] = convolution

    i_transformed_ar.append(i_transformed)
    # Plot the image. Note the size of the axes -- they are 512 by 512
    plt.title('Conv2D')
    plt.gray()
    plt.grid(False)
    plt.imshow(i_transformed_ar[it])
    it = it + 1
    # plt.axis('off')
    plt.show()

for _it in range(0,it):
    new_x = int(size_x / 2)
    new_y = int(size_y / 2)
    newImage = np.zeros((new_x, new_y))
    for x in range(0, size_x, 2):
        for y in range(0, size_y, 2):
            pixels = []
            pixels.append(i_transformed_ar[_it][x, y])
            pixels.append(i_transformed_ar[_it][x + 1, y])
            pixels.append(i_transformed_ar[_it][x, y + 1])
            pixels.append(i_transformed_ar[_it][x + 1, y + 1])
            newImage[int(x / 2), int(y / 2)] = max(pixels)

    # Plot the image. Note the size of the axes -- now 256 pixels instead of 512
    plt.title('MaxPooling2D')
    plt.gray()
    plt.grid(False)
    plt.imshow(newImage)
    # plt.axis('off')
    plt.show()

