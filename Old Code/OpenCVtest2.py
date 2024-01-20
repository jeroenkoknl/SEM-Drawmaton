import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

def helper_ImageToXY(imgfile, L1, L2, L3):

    img = cv2.imread(imgfile, cv2.IMREAD_GRAYSCALE)
    # Create a binary version of the img to improve quality of contour detection
    img2 = cv2.imread(imgfile, cv2.IMREAD_COLOR)
    _, threshold = cv2.threshold(img, 117, 255, cv2.THRESH_BINARY)

    # Use image dimensions and linkage's max span to calculate a scale from pixels to centimeters
    # via equating the image diagonal length to the maximum spanning length of the linkages
    height = img.shape[0]
    width = img.shape[1]
    maxlinkagespan = L2 + L3

    # Create contours, and find minimum enclosing rectangle's center coordinates
    contours, _ = cv2.findContours(
        threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    print(len(contours))
    correctcontour = contours[0]
    for i in range(len(contours)):
        cv2.drawContours(img2, contours, i, (0, 0, 255), 5)
        name = "contour " + str(i)
        cv2.imshow(name, img2)
        print("is this the right line drawing? (y/n)")
        if cv2.waitKey(0) & 0xFF == ord('n'):
            cv2.destroyWindow(name)
            print('oops')
            img2 = cv2.imread(imgfile, cv2.IMREAD_COLOR)
        else:
            correctcontour = contours[i]
            print("gottem")
            cv2.destroyWindow(name)
            break
        
    pixpercm = ((height**2 + width**2)/(maxlinkagespan**2))**0.5
    minrectcenter, minrectdims, minrectangle = cv2.minAreaRect(correctcontour)
    print(minrectcenter, minrectdims, minrectangle)
    minrectcx = minrectcenter[0]
    minrectcy = height - minrectcenter[1]
    
    rx, ry, rw, rh = cv2.boundingRect(correctcontour)
    
    # print(rx, ry, rw, rh)
    # Unravel the array of (x,y) coordinates into an array of alternating x and y vals [x1, y1, x2, y2, ...]
    contour = np.ravel(correctcontour)
    # Calculate how many x and y pairs to store, by finding the arclength of the contour in cm,
    # and determing step, the number of indices between the x,y pairs to be sampled from the contour array
    contour_arclen = len(contour)/pixpercm
    # step = 2*int(0.5*len(contour)/contour_arclen)
    step = 2*int(len(contour)/2/contour_arclen)
    print(len(contour), contour_arclen, pixpercm, step)
    if (step == 0): step = 2
    # Establish tvals, an array representing the time at which each x,y pair is drawn
    tvals = np.arange(0, len(contour), step)
    # print(contour, len(contour))
    tvals = np.append(tvals, len(contour) - 2)
    # print(tvals, len(tvals))
    xcoords = np.zeros(len(tvals))
    ycoords = np.zeros(len(tvals))

    # Iterate through tvals and adjust each pixel in the x & y directions, then convert from pixels to cm
    i = 0
    for t in tvals:
        # x = (contour[t] + xadjust) / pixpercm
        # y = (height - contour[t+1] + yadjust) / pixpercm
        # print(t, contour[t], contour[t+1], i)
        # if (t == len(contour) - 1):
        #     print(t, contour[t], print(i+1))
        x = contour[t] / pixpercm
        y = (height - contour[t+1]) / pixpercm
        xcoords[i] = x
        ycoords[i] = y
        i += 1

    
    # for i in range (len(contour)):
    #     t = tvals[i]
    #     x = contour[t] / pixpercm
    #     y = (height - contour[t+1]) / pixpercm
    #     xcoords[i] = x
    #     ycoords[i] = y

    return [xcoords, ycoords, rx/pixpercm, (height-ry)/pixpercm, rw/pixpercm, rh/pixpercm]    

def PositionImage(xcoords, ycoords, brectx, brecty, brectw, brecth, targetx, targety, targetw, targeth):
    wscale = targetw / brectw
    hscale = targeth / brecth
    if (wscale <= hscale):
        print('hi') 
        xcoords,ycoords = scale(wscale, xcoords, ycoords, brectw, brecth)
    else:
        xcoords,ycoords = scale(hscale, xcoords, ycoords, brectw, brecth)
        
    xcoords += targetx - np.min(xcoords)
    ycoords += targety - np.max(ycoords)
    
    return xcoords, ycoords
    
def scale(factor, xcoords, ycoords, currw, currh):
    neww = currw*factor
    newh = currh*factor
    for i in range(len(xcoords)):
        xcoords[i] *= factor
        ycoords[i] *= factor
    return [xcoords, ycoords]    
   

def ImageToXY(L1, L2, L3, Gx, Gy, imgfile, targetx, targety, targetw, targeth):

    fig, ax = plt.subplots()
    
    xcoords, ycoords, brectx, brecty, brectw, brecth = helper_ImageToXY(imgfile, L1, L2, L3)
    # ax.plot(xcoords, ycoords)
    print(brectx, brecty, brectw, brecth)

    adjxcoords, adjycoords = PositionImage(xcoords, ycoords, brectx, brecty, brectw, brecth, targetx, targety, targetw, targeth)
    
    ax.plot(adjxcoords, adjycoords)
    ax.scatter([0,Gx], [0, Gy])
    refcirc = plt.Circle((0, 0), radius=L2 + L3, fill=False)
    brect = Rectangle((brectx,brecty), brectw, -brecth, color="orange", fill=False)
    adjrect = Rectangle((targetx, targety), targetw, -targeth, color="green", fill=False)
    ax.add_patch(brect)
    ax.add_patch(adjrect)
    ax.set_xlim(-20, 50)
    ax.set_ylim(-20, 50)
    ax.set_aspect('equal')
    plt.gca().add_artist(refcirc)
    plt.grid()
    plt.show()
    return adjxcoords, adjycoords


L1 = 4.1
L2 = 16.2
L3 = 24.0
Gx = -4.0
Gy = 15.0
imgfile = "./InputImages/looptest.png"
targetx = 13
targety = 26
targetw = 16
targeth = 16 
xcoords, ycoords = ImageToXY(L1, L2, L3, Gx, Gy, imgfile, targetx, targety, targetw, targeth)
print(xcoords, ycoords)
print(len(xcoords))

# # fig, ax = plt.subplots()

# # img = cv2.imread('Car.jpg', cv2.IMREAD_GRAYSCALE)
# # img2 = cv2.imread('Car.jpg', cv2.IMREAD_COLOR)
# # _, threshold = cv2.threshold(img, 117, 255, cv2.THRESH_BINARY)

# # height = img.shape[0]
# # width = img.shape[1]

# # contours, _ = cv2.findContours(
# #     threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
# # correctcontour = contours[0]
# # L2 = 15.8
# # L3 = 23.4
# # maxspan = L2 + L3

# # for i in range(len(contours)):
# #     cv2.drawContours(img2, contours, i, (0, 0, 255), 5)
# #     name = "contour " + str(i)
# #     cv2.imshow(name, img2)
# #     print("is this the right line drawing? (y/n)")

# #     if cv2.waitKey(0) & 0xFF == ord('n'):
# #         cv2.destroyWindow(name)
# #         img2 = cv2.imread('CarFront.jpg', cv2.IMREAD_COLOR)
# #     else:
# #         correctcontour = contours[i]
# #         print("gottem")
# #         cv2.destroyWindow(name)
# #         break

# # minrectcx, minrectcy, minrectw, minrecth = cv2.boundingRect(correctcontour)
# # pixpercm = ((minrecth**2 + minrectw**2)/(maxspan**2))**0.5
# # minrectcy = height - minrectcy
# # minrecth = height - minrecth
# # ax.add_patch(Rectangle((minrectcx/pixpercm, minrectcy/pixpercm),
# #              minrectw/pixpercm, minrecth/pixpercm, fill=False))
# # ax.plot(correctcontour[0], correctcontour[1])
# # contour = np.ravel(correctcontour)


# # contour_arclen = len(contour)/pixpercm
# # step = 2*int(len(contour)/2/contour_arclen)
# # print(pixpercm, minrectcx/pixpercm, minrectcy/pixpercm, contour_arclen, step)

# # xadjust = minrectcx - minrectw
# # yadjust = minrectcy - minrecth

# # # xadjust = (width - minrectcx) / 2
# # # yadjust = (height - minrectcy) / 2
# # # print(contour[0], height - contour[1], xadjust/pixpercm, yadjust/pixpercm)

# # tvals = np.arange(0, len(contour), step)
# # tvals = np.append(tvals, len(contour) - 2)
# # xcoords = np.zeros(len(tvals))
# # ycoords = np.zeros(len(tvals))

# # i = 0
# # for t in tvals:
# #     x = (contour[t] + xadjust) / pixpercm
# #     y = (height - contour[t+1] + yadjust) / pixpercm
# #     # print(t, x, y)
# #     xcoords[i] = x
# #     ycoords[i] = y
# #     i += 1

# # print(len(xcoords), len(ycoords))
# # # print("\n\n", ycoords)
# # # # Showing the final image.
# # cv2.imshow('image2', img2)

# # L3 = 23.4
# # t = np.linspace(0, 2*np.pi, 75, dtype=float)

# # ax.plot(xcoords, ycoords)
# # ax.plot(maxspan*np.cos(t), maxspan*np.sin(t))
# # ax.plot((minrectcx)/pixpercm, (minrectcy)/pixpercm, "o", color='red')

# # ax.plot((minrectcx + xadjust)/pixpercm, (minrectcy + yadjust)/pixpercm, "o", color='green')
# # ax.set_aspect('equal')

# # plt.show()
# # print("hi")
# # # # Exiting the window if 'q' is pressed on the keyboard.
# # if cv2.waitKey(0) & 0xFF == ord('q'):
# #     cv2.destroyAllWindows()


# import numpy as np
# import cv2
# import matplotlib.pyplot as plt
# from matplotlib.patches import Rectangle

# fig, ax = plt.subplots()

# img = cv2.imread('blackcat.jpg', cv2.IMREAD_GRAYSCALE)
# img2 = cv2.imread('blackcat.jpg', cv2.IMREAD_COLOR)
# _, threshold = cv2.threshold(img, 117, 255, cv2.THRESH_BINARY)

# height = img.shape[0]
# width = img.shape[1]

# contours, _ = cv2.findContours(
#     threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
# correctcontour = contours[0]
# L2 = 16.2
# L3 = 24
# maxspan = L2 + L3

# for i in range(len(contours)):
#     cv2.drawContours(img2, contours, i, (0, 0, 255), 5)
#     name = "contour " + str(i)
#     cv2.imshow(name, img2)
#     print("is this the right line drawing? (y/n)")

#     if cv2.waitKey(0) & 0xFF == ord('n'):
#         cv2.destroyWindow(name)
#         img2 = cv2.imread('CarFront.jpg', cv2.IMREAD_COLOR)
#     else:
#         correctcontour = contours[i]
#         print("gottem")
#         cv2.destroyWindow(name)
#         break
# # ax.plot(width, height, "o")
# minrectcx, minrectcy, minrectw, minrecth = cv2.boundingRect(correctcontour)
# # minrectcy = height - minrectcy
# # minrecth = height - minrecth
# # ax.add_patch(Rectangle((minrectcx, minrectcy),
# #  minrectw, minrecth, fill=False))
# # pixpercm = 2*((minrecth**2 + minrectw**2)/(maxspan**2))**0.5
# pixpercm = (((minrecth + minrectcy)**2 +
#             (minrectw + minrectcx)**2)/(maxspan**2))**0.5


# contour = np.ravel(correctcontour)
# print(contour)
# print(len(contour))
# contour_arclen = len(contour)/pixpercm
# step = 2*int(len(contour)/contour_arclen)

# # Establish tvals, an array representing the time at which each x,y pair is drawn
# tvals = np.arange(0, len(contour), step)
# tvals = np.append(tvals, len(contour) - 2)
# xcoords = np.zeros(len(tvals))
# ycoords = np.zeros(len(tvals))
# print(len(xcoords))
# i = 0
# for t in tvals:
#     x = (contour[t]) / pixpercm
#     y = (contour[t+1]) / pixpercm
#     xcoords[i] = x
#     ycoords[i] = y
#     i += 1


# ax.plot(width/pixpercm, height/pixpercm, "o")
# ax.plot(xcoords, ycoords)
# ax.add_patch(Rectangle((minrectcx/pixpercm, minrectcy/pixpercm),
#                        minrectw/pixpercm, minrecth/pixpercm, fill=False))

# t = np.linspace(0, 2*np.pi, 50)
# ax.plot(maxspan*np.cos(t), maxspan*np.sin(t))
# ax.plot(0, 0, "o")
# # ax.plot(xcoords, ycoords)
# ax.set_aspect('equal')
# plt.show()
