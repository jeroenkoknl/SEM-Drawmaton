import numpy as np
from sympy import *
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import cv2

def helper_ImageToXY(imgfile, L1, L2, L3, interactive=True):

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
    correctcontour = contours[0]
    if interactive:
        for i in range(len(contours)):
            cv2.drawContours(img2, contours, i, (0, 0, 255), 5)
            name = "contour " + str(i)
            cv2.imshow(name, img2)
            print("is this the right line drawing? (y/n)")

            if cv2.waitKey(0) & 0xFF == ord('n'):
                cv2.destroyWindow(name)
            else:
                correctcontour = contours[i]
                print("Selected contour", i)
                cv2.destroyWindow(name)
                break
    else:
        # In non-interactive mode, select the largest contour
        max_area = 0
        for i, contour in enumerate(contours):
            area = cv2.contourArea(contour)
            if area > max_area:
                max_area = area
                correctcontour = contour
        print("Automatically selected largest contour")
        
    pixpercm = ((height**2 + width**2)/(maxlinkagespan**2))**0.5
    minrectcenter, minrectdims, minrectangle = cv2.minAreaRect(correctcontour)
    # print(minrectcenter, minrectdims, minrectangle)
    minrectcx = minrectcenter[0]
    minrectcy = height - minrectcenter[1]
    
    rx, ry, rw, rh = cv2.boundingRect(correctcontour)
    
    # print(rx, ry, rw, rh)
    # Unravel the array of (x,y) coordinates into an array of alternating x and y vals [x1, y1, x2, y2, ...]
    contour = np.ravel(contours[0])
    # Calculate how many x and y pairs to store, by finding the arclength of the contour in cm,
    # and determing step, the number of indices between the x,y pairs to be sampled from the contour array
    contour_arclen = len(contour)/pixpercm
    step = 2*int(len(contour)/6/contour_arclen)
    if (step == 0): step = 2
    
    # Establish tvals, an array representing the time at which each x,y pair is drawn
    tvals = np.arange(0, len(contour), step)
    tvals = np.append(tvals, len(contour) - 2)
    xcoords = np.zeros(len(tvals))
    ycoords = np.zeros(len(tvals))

    # Iterate through tvals and adjust each pixel in the x & y directions, then convert from pixels to cm
    i = 0
    for t in tvals:
        # x = (contour[t] + xadjust) / pixpercm
        # y = (height - contour[t+1] + yadjust) / pixpercm
        x = contour[t] / pixpercm
        y = (height - contour[t+1]) / pixpercm
        xcoords[i] = x
        ycoords[i] = y
        i += 1

    return [xcoords, ycoords, rx/pixpercm, (height-ry)/pixpercm, rw/pixpercm, rh/pixpercm]    

def PositionImage(xcoords, ycoords, brectx, brecty, brectw, brecth, targetx, targety, targetw, targeth):
    wscale = targetw / brectw
    hscale = targeth / brecth
    if (wscale <= hscale):
        # print('hi') 
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
   

def ImageToXY(L1, L2, L3, Gx, Gy, imgfile, targetx, targety, targetw, targeth, interactive=True, preview_mode='none'):
    print("2. Processing input image...")
    print("   - Reading image file")
    xcoords, ycoords, brectx, brecty, brectw, brecth = helper_ImageToXY(imgfile, L1, L2, L3, interactive)
    adjxcoords, adjycoords = PositionImage(xcoords, ycoords, brectx, brecty, brectw, brecth, targetx, targety, targetw, targeth)
    
    import utilities as util
    util.show_preview(
        coords=(adjxcoords, adjycoords),
        dims=(L1, L2, L3, Gx, Gy),
        title="Image Preview",
        preview_mode=preview_mode,
        close_delay=3
    )
    
    return adjxcoords, adjycoords

