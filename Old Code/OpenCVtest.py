import numpy as np
import cv2
import matplotlib.pyplot as plt

img = cv2.imread('linecrosstest.png', cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread('linecrosstest.png', cv2.IMREAD_COLOR)
_, threshold = cv2.threshold(img, 117, 255, cv2.THRESH_BINARY)

height = img.shape[0]
width = img.shape[1]

contours, _ = cv2.findContours(
    threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
correctcontour = contours[0]
L2 = 15.8
L3 = 23.4
maxspan = L2 + L3

for i in range(len(contours)):
    cv2.drawContours(img2, contours, i, (0, 0, 255), 5)
    name = "contour " + str(i)
    cv2.imshow(name, img2)
    print("is this the right line drawing? (y/n)")

    if cv2.waitKey(0) & 0xFF == ord('n'):
        cv2.destroyWindow(name)
        img2 = cv2.imread('CarFront.jpg', cv2.IMREAD_COLOR)
    else:
        correctcontour = contours[i]
        print("gottem")
        cv2.destroyWindow(name)
        break

minrectcenter, minrectdims, minrectangle = cv2.minAreaRect(correctcontour)
minrectcx = minrectcenter[0]
minrectcy = minrectcenter[1]

contour = np.ravel(correctcontour)
# print(len(contour))
pixpercm = ((height**2 + width**2)/(maxspan**2))**0.5
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

L3 = 23.4
t = np.linspace(0, 2*np.pi, 75, dtype=float)

fig, ax = plt.subplots()
ax.plot(xcoords, ycoords)
ax.plot(maxspan*np.cos(t), maxspan*np.sin(t))
ax.plot((minrectcx)/pixpercm, (minrectcy)/pixpercm, "o")

ax.plot((minrectcx + xadjust)/pixpercm, (minrectcy + yadjust)/pixpercm, "o")
ax.set_aspect('equal')

plt.show()
print("hi")
# # Exiting the window if 'q' is pressed on the keyboard.
if cv2.waitKey(0) & 0xFF == ord('q'):
    cv2.destroyAllWindows()
