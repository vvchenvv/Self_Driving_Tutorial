import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from PIL import Image
from numpy import array

# Read in the image and print out some stats
# image = mpimg.imread('ColorSelectionTest.png')
image_P = Image.open('ColorSelectionTest.jpg')
print('This image is: ',type(image_P), 
         'with dimensions:', image_P.size)

image=array(image_P)

# Pull out the x and y sizes and make a copy of the image
ysize = image.shape[0]
xsize = image.shape[1]
region_color_select = np.copy(image)
line_select = np.copy(image)

left_bottom = [70,540]
right_bottom = [860,540]
apex = [480, 280]

# Fit lines (y=Ax+B) to identify the  3 sided region of interest
# np.polyfit() returns the coefficients [A, B] of the fit
fit_left = np.polyfit((left_bottom[0], apex[0]), (left_bottom[1], apex[1]), 1)
fit_right = np.polyfit((right_bottom[0], apex[0]), (right_bottom[1], apex[1]), 1)
fit_bottom = np.polyfit((left_bottom[0], right_bottom[0]), (left_bottom[1], right_bottom[1]), 1)

#Color threshold
red_threshold = 220
green_threshold = 220
blue_threshold = 220
rgb_threshold = [red_threshold, green_threshold, blue_threshold]

# Identify pixels below the threshold
thresholds = (image[:,:,0] < rgb_threshold[0]) \
            | (image[:,:,1] < rgb_threshold[1]) \
            | (image[:,:,2] < rgb_threshold[2])

# Find the region inside the lines
XX, YY = np.meshgrid(np.arange(0, xsize), np.arange(0, ysize))
region_thresholds = (YY > (XX*fit_left[0] + fit_left[1])) & \
                    (YY > (XX*fit_right[0] + fit_right[1])) & \
                    (YY < (XX*fit_bottom[0] + fit_bottom[1]))

# Color pixels red which are inside the region of interest

region_color_select[thresholds]=[0,0,0]
region_color_select[~region_thresholds] = [0, 0, 0]
# region_color_select[~thresholds & region_thresholds]=[255,0,0]
# region_color_select[~thresholds&region_color_select]=[255,0,0]
# print(region_thresholds)

# Display the image
plt.imshow(region_color_select)
plt.show()