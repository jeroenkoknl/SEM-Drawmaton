import numpy as np
import skimage.morphology
import cv2
from skan import Skeleton
import matplotlib.pyplot as plt


input_image = cv2.imread('circletest.jpg', cv2.IMREAD_GRAYSCALE)

num_labels, labels_im = cv2.connectedComponents(input_image)
skeleton_image = skimage.morphology.skeletonize(labels_im == 1)
# skeleton = Skeleton(skeleton_image)
# path_coordinates = skeleton.coordinates[skeleton.path(0)]
# print(path_coordinates.shape)
# print(path_coordinates)

# display results
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(8, 4),
                         sharex=True, sharey=True)

ax = axes.ravel()

ax[0].imshow(input_image, cmap=plt.cm.gray)
ax[0].axis('off')
ax[0].set_title('original', fontsize=20)

ax[1].imshow(skeleton_image, cmap=plt.cm.gray)
ax[1].axis('off')
ax[1].set_title('skeleton', fontsize=20)

fig.tight_layout()
plt.show()
