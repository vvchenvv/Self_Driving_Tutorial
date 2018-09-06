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

print(len(image))
print(len(image[0]))
print(len(image[0][0]))
print("-------len--------")
print(image[0])

# Grab the x and y size and make a copy of the image
ysize = image.shape[0]
xsize = image.shape[1]
# Note: always make a copy rather than simply using "="
color_select = np.copy(image)

# print(color_select[1])

# Define our color selection criteria
# Note: if you run this code, you'll find these are not sensible values!!
# But you'll get a chance to play with them soon in a quiz
red_threshold = 180
green_threshold = 180
blue_threshold = 180
rgb_threshold = [red_threshold, green_threshold, blue_threshold]

# Identify pixels below the threshold
thresholds = (image[:,:,0] < rgb_threshold[0]) \
            | (image[:,:,1] < rgb_threshold[1]) \
            | (image[:,:,2] < rgb_threshold[2])

print(len(thresholds))
print(len(thresholds[0]))


color_select[thresholds] = [0,0,0]

# Display the image                 
plt.imshow(color_select)
plt.show()