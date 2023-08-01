import numpy as np
import cv2
import matplotlib.pyplot as plt

img = cv2.imread('TulipTest.jpg', cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread('TulipTest.jpg', cv2.IMREAD_COLOR)
_, threshold = cv2.threshold(img, 117, 255, cv2.THRESH_BINARY)

height = img.shape[0]
width = img.shape[1]

contours, _ = cv2.findContours(
    threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

cv2.drawContours(img2, contours, -1, (0, 0, 255), 5)

minrectcenter, minrectdims, minrectangle = cv2.minAreaRect(contours[0])
minrectcx = minrectcenter[0]
minrectcy = minrectcenter[1]

L2 = 15.8
contour = np.ravel(contours[0])
# print(len(contour))
pixpercm = ((height**2 + width**2)/(4*L2**2))**0.5
contour_arclen = len(contour)/pixpercm
step = 2*int(len(contour)/2/contour_arclen)
print(pixpercm, minrectcx/pixpercm, (height -
      minrectcy)/pixpercm, contour_arclen, step)

minrectcy = height - minrectcy
xadjust = (width - minrectcx) / 2
yadjust = (height - minrectcy) / 2
print(contour[0], height - contour[1], xadjust/pixpercm, yadjust/pixpercm)

tvals = np.arange(0, len(contour), step)
tvals = np.append(tvals, len(contour) - 2)
xcoords = np.zeros(len(tvals))
ycoords = np.zeros(len(tvals))

i = 0
for t in tvals:
    x = (contour[t] + xadjust) / pixpercm
    y = (height - contour[t+1] + yadjust) / pixpercm
    # print(t, x, y)
    xcoords[i] = x
    ycoords[i] = y
    i += 1

print(len(xcoords), len(ycoords))
# print("\n\n", ycoords)
# # Showing the final image.
cv2.imshow('image2', img2)

fig, ax = plt.subplots()
ax.plot(xcoords, ycoords)
ax.set_aspect('equal')

plt.show()

# # Exiting the window if 'q' is pressed on the keyboard.
if cv2.waitKey(0) & 0xFF == ord('q'):
    cv2.destroyAllWindows()
