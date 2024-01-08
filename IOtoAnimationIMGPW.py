import numpy as np
import sympy as sp
from sympy import *
from sympy.parsing.sympy_parser import parse_expr
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import cv2

# hello test commit

def StoreDims(file):
    # Gives user the option to use standard linkage dimensions and rotor base position to save time
    print("use default dimensions?")
    if str(input()) == "yes":
        L1 = 7.6
        L2 = 15.8
        L3 = L1 + L2
        Gx = -15
        Gy = 7.6
    else:
        # Prompts for and saves each dimension value
        print("input L1 value: ")
        L1 = float(input())
        print("input L2 value: ")
        L2 = float(input())
        print("input L3 value: ")
        L3 = float(input())
        print("input Gx value: ")
        Gx = float(input())
        print("input Gy value: ")
        Gy = float(input())
    # Write all dimensions to the storage file
    file.write(str(L1) + "\n")
    file.write(str(L2) + "\n")
    file.write(str(L3) + "\n")
    file.write(str(Gx) + "\n")
    file.write(str(Gy) + "\n")
    return np.array([L1, L2, L3, Gx, Gy])


def StoreParametricEQs(file):
    # Function that determines how to read, store, and write parametric equations
    targetxfuncs = []
    targetyfuncs = []
    # Calls the corresponding helper function to read parametric equations from a txt file or via manual IO
    print("read parametric functions from txt file (yes), or input functions manually (no)")
    if (str(input()) == "yes"):
        return ReadStoredPieceWiseFuncs(file)
    print("how many pieces? (at most 20, if continuous, enter 1)")
    # store the number of pieces
    piececount = int(input())
    file.write(str(piececount) + "\n")
    starts = np.zeros(piececount)
    ends = np.zeros(piececount)
    # Depending on the number of pieces, conducts the IO to manually store the parametric functions
    if piececount > 1:
        print("Piece-wise mode: make sure that the starting time is 0 and ending time is 6.2831...(2*pi)")
    for i in range(piececount):
        # Prompts for and stores the x and y functions for each piece, i from 1 to piece count
        print("input x" + str(i+1) +
              "(t), the function for the pen path's x position: ")
        xinput = str(input())
        print("input y" + str(i+1) +
              "(t), the function for the pen path's y position: ")
        yinput = str(input())

        # Forces the domain to be 0 to 2*pi if there is only 1 x,y function pair
        if piececount == 1:
            startinput = 0
            endinput = 2*np.pi
        else:
            print("input the starting tval for this XY pair")
            startinput = float(input())
            print("input the ending tval for this XY pair")
            endinput = float(input())

        # Write x,y functions and start,end pairs to file
        file.write(xinput + ", " + yinput + ", " +
                   str(startinput) + ", " + str(endinput) + "\n")
        # Stores the x,y function pairs, and start,end pairs for local use using sympy's parse_expr to convert from str to sympy expression
        targetxfuncs.append(parse_expr(xinput))
        targetyfuncs.append(parse_expr(yinput))
        starts[i] = startinput
        ends[i] = endinput
    return targetxfuncs, targetyfuncs, starts, ends


def ReadStoredPieceWiseFuncs(wfile):
    # Gives the option to read piece-wise parametric functions from a file, since inputing them manually is error-prone
    print("enter file name for stored piecewise functions:")
    rfile = str(input())
    # Reads the x,y functions and start,end pairs from the read file
    xfuncs = np.loadtxt(rfile, delimiter=",", usecols=0, dtype=str)
    yfuncs = np.loadtxt(rfile, delimiter=",", usecols=1, dtype=str)
    starts = np.loadtxt(rfile, delimiter=",",
                        usecols=2, dtype=float)
    ends = np.loadtxt(rfile, delimiter=",", usecols=3, dtype=float)
    pieces = len(xfuncs)
    # Store the piece count
    wfile.write(str(pieces) + "\n")
    targetxfuncs = []
    targetyfuncs = []
    # Store the x,y function pairs and the domain start,end pairs, and using sympy's parse_expr to convert strings into sympy equations
    for i in range(len(xfuncs)):
        wfile.write(xfuncs[i] + ", " + yfuncs[i] + ", " +
                    str(starts[i]) + ", " + str(ends[i]) + "\n")
        targetxfuncs.append(parse_expr(xfuncs[i]))
        targetyfuncs.append(parse_expr(yfuncs[i]))
    return targetxfuncs, targetyfuncs, starts, ends


def ParametricToXY(file):
    # Get the x,y parametric function pairs, and their corresponding domains
    targetxfuncs, targetyfuncs, starts, ends = StoreParametricEQs(file)
    t = symbols('t')
    # Choose the number of data points based on the number of x,y parametric pairs
    datacount = 120 + 5*len(targetxfuncs)
    tvals = np.linspace(0, 2*np.pi, datacount)
    xcoords = np.zeros(len(tvals))
    ycoords = np.zeros(len(tvals))
    # Keep track of which x,y functions to evaluate depending on which domain the current tval lies in
    funcind = 0
    for i in range(len(tvals)):
        tval = tvals[i]
        # Increment the current function if we have entered the succeeding domain
        if tval >= ends[funcind] and funcind != len(ends) - 1:
            funcind += 1
        xcoords[i] = targetxfuncs[funcind].subs(t, tval)
        ycoords[i] = targetyfuncs[funcind].subs(t, tval)
    return [xcoords, ycoords]


def ImageToXY(file, L1, L2, L3):
    # Use openCV library to process an image
    print("enter path for image")
    imgpath = str(input())
    # Put the image path into the storage file, and the piece-count as 1
    file.write(str(1) + "\n" + "image from " + imgpath + "\n")
    print("processing image")
    img = cv2.imread(imgpath, cv2.IMREAD_GRAYSCALE)
    # Create a binary version of the img to improve quality of contour detection
    img2 = cv2.imread(imgpath, cv2.IMREAD_COLOR)
    _, threshold = cv2.threshold(img, 117, 255, cv2.THRESH_BINARY)

    # Use image dimensions and linkage's max span to calculate a scale from pixels to centimeters
    # via equating the image diagonal length to the maximum spanning length of the linkages
    height = img.shape[0]
    width = img.shape[1]
    maxlinkagespan = L2 + L3
    pixpercm = ((height**2 + width**2)/(maxlinkagespan**2))**0.5

    # Create contours, and find minimum enclosing rectangle's center coordinates
    contours, _ = cv2.findContours(
        threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    minrectcenter, minrectdims, minrectangle = cv2.minAreaRect(contours[0])
    minrectcx = minrectcenter[0]
    minrectcy = height - minrectcenter[1]
    # Calculate the x and y adjustments to force the center of the contour's minimum enclosing rectangle to align w/ the canvas center
    xadjust = (width - minrectcx) / 2
    yadjust = (height - minrectcy) / 2

    # Unravel the array of (x,y) coordinates into an array of alternating x and y vals [x1, y1, x2, y2, ...]
    contour = np.ravel(contours[0])
    # Calculate how many x and y pairs to store, by finding the arclength of the contour in cm,
    # and determing step, the number of indices between the x,y pairs to be sampled from the contour array
    contour_arclen = len(contour)/pixpercm
    step = 2*int(len(contour)/2/contour_arclen)

    # Establish tvals, an array representing the time at which each x,y pair is drawn
    tvals = np.arange(0, len(contour), step)
    tvals = np.append(tvals, len(contour) - 2)
    xcoords = np.zeros(len(tvals))
    ycoords = np.zeros(len(tvals))

    # Iterate through tvals and adjust each pixel in the x & y directions, then convert from pixels to cm
    i = 0
    for t in tvals:
        x = (contour[t] + xadjust) / pixpercm
        y = (height - contour[t+1] + yadjust) / pixpercm
        xcoords[i] = x
        ycoords[i] = y
        i += 1

    return [xcoords, ycoords]


def CalcThetaVals(file, L1, L2, L3):
    # Main function to calculate the theta 1 and 2 values necessary
    # Use IO to determine the source of the intended pen path
    # Convert the pen path into X,Y coordinates
    print("Obtain target line path from image? (yes/no)")
    if (str(input()) == "yes"):
        XYcoords = ImageToXY(file, L1, L2, L3)
    else:
        print(
            "Obtain target line path from parametric equations? (yes/no)")
        if (str(input()) == "yes"):
            XYcoords = ParametricToXY(file)

    print("Calculating theta values, please wait")
    # Open up raw data files for raw values from sympy solver, and call a cleaning function to choose the apropriate values
    rawname = "Raw" + file.name
    rawfile = open(rawname, "w")
    CalcRawThetaValsXY(rawfile, L1, L2, L3, XYcoords[0], XYcoords[1])
    return CleanThetaVals(rawfile.name, file)


def CalcRawThetaValsXY(rawfile, L1, L2, L3, xcoords, ycoords):
    # Turns the target pen path's X,Y coordinates into theta1 and theta2 values
    # based on the linkage dimensions, using sympy's solver
    delim = " "
    th1, th2 = symbols('th1, th2', real=True)
    # Iterate through all x,y coordinates, calculate, and store both sympy solutions in a raw data file
    for i in range(len(xcoords)):
        eqx = Eq(L2*cos(th2) - L3*sin(th1), xcoords[i])
        eqy = Eq(L2*sin(th2) + L3*cos(th1), ycoords[i])
        soln = solve((eqx, eqy), (th1, th2))
        print(soln, "\n")
        rawfile.write(str(soln[0][0]) + delim + str(soln[0][1]) +
                      delim + str(soln[1][0]) + delim + str(soln[1][1]) + "\n")
    rawfile.close()


def CleanThetaVals(rawfile, file):
    # Clean the theta values from the rawfile by choosing the correct values from the two possible solution pairs
    print("Cleaning theta values")
    # The theta 1 and 2 values from the 0th and 1st solution pairs that sympy solve generates
    rS0theta1 = np.loadtxt(rawfile, usecols=0, dtype=float)
    rS0theta2 = np.loadtxt(rawfile, usecols=1, dtype=float)
    rS1theta1 = np.loadtxt(rawfile, usecols=2, dtype=float)
    rS1theta2 = np.loadtxt(rawfile, usecols=3, dtype=float)

    # The final solution theta1 and theta2 arrays
    theta1 = np.zeros(len(rS0theta1))
    theta2 = np.zeros(len(rS0theta2))
    for i in range(len(theta1)):
        # Choose the 1st solution pair by default, or if the previous theta values were within a 0.6 radian margin
        # from the current 1st solution theta values
        if (i == 0 or (np.abs(rS1theta1[i] - theta1[i-1]) <= 0.6 and np.abs(rS1theta2[i] - theta2[i-1]) <= 0.6)):
            theta1[i] = rS1theta1[i]
            theta2[i] = rS1theta2[i]
        else:
            # Choose the 0th solution pair theta values instead
            theta1[i] = rS0theta1[i]
            theta2[i] = rS0theta2[i]
        file.write(str(theta1[i]) + " " + str(theta2[i]) + "\n")
    print("finished clean")
    file.close()
    return theta1, theta2


def CalcRadiiVals(L1, Gx, Gy, theta1, theta2):
    # a is a parametric angle variable in radians
    a = np.linspace(0, 2*np.pi, len(theta1), dtype=float)
    # b is the angle relating the center of the rotors to the corresponding linkage point of contact
    # there is a b for the top and bottom rotors
    b_top = np.arctan2(-Gy + L1*np.cos(theta1), -Gx - L1*np.sin(theta1))
    c_top = -a + b_top
    b_bottom = np.arctan2(-Gy + L1*np.cos(theta2), -Gx - L1*np.sin(theta2))
    c_bottom = -a + b_bottom

    # The radii for the top and bottom rotors is the norm of the vector from the rotor base @ (Gx, Gy)
    # to the top and bottom linkage contact points
    r_top = ((-Gx - L1*np.sin(theta1))**2 + (-Gy + L1*np.cos(theta1))**2)**0.5
    r_bottom = ((-Gx - L1*np.sin(theta2))**2 +
                (-Gy + L1*np.cos(theta2))**2)**0.5
    return a, r_bottom, r_top, c_bottom, c_top


def animate(t, ax, L1, L2, L3, Gx, Gy, theta1, theta2, a, r_bottom, r_top, c_bottom, c_top):
    # Save frame number t for later use
    i = t
    # Convert t into an angle value in radians
    t = a[i % len(a)]

    # Clear the frame
    ax.clear()
    # Plot the origin and the rotor base position
    ax.plot(0, 0, "o", color="black")
    ax.plot(Gx, Gy, ".", color="black")

    # Plot objects & paths:
    # pen path
    ax.plot(L2*np.cos(theta2) - L3*np.sin(theta1),
            L2*np.sin(theta2) + L3*np.cos(theta1), color="red")
    # top rotor
    ax.plot(r_top*np.cos(c_top)*np.cos(t) - r_top*np.sin(c_top)*np.sin(t) + Gx,
            r_top*np.cos(c_top)*np.sin(t) + r_top*np.sin(c_top)*np.cos(t) + Gy, color="green")
    # bottom rotor
    ax.plot(r_bottom*np.cos(c_bottom)*np.cos(t) - r_bottom*np.sin(c_bottom)*np.sin(t) + Gx,
            r_bottom*np.cos(c_bottom)*np.sin(t) + r_bottom*np.sin(c_bottom)*np.cos(t) + Gy, color="blue")

    # Convert t into an index to access the corresponding theta value at the current frame/time
    t = i % len(a)

    # Plot moving points:
    # pen position
    ax.plot(L2*cos(theta2[t]) - L3*sin(theta1[t]), L2 *
            sin(theta2[t]) + L3*cos(theta1[t]), ".", color='yellow')
    # top linkage contact point
    ax.plot(-L1*sin(theta1[t]), L1*cos(theta1[t]), ".", color='green')
    # bottom linkage contact point
    ax.plot(-L1*sin(theta2[t]), L1*cos(theta2[t]), ".", color="blue")

    # top linkage (linkage 4 in CAD)
    ax.plot([0, -L1*sin(theta1[t])], [0, L1*cos(theta1[t])], color="red")
    # bottom linkage (linkage 1 in CAD)
    ax.plot([0, -L1*sin(theta2[t])], [0, L1*cos(theta2[t])], color="orange")
    ax.plot([0, L2*cos(theta2[t])], [0, L2*sin(theta2[t])], color="orange")
    # linkage 2 in CAD
    ax.plot([-L1*sin(theta1[t]), -L1*sin(theta1[t]) + L2*cos(theta2[t])],
            [L1*cos(theta1[t]), L1*cos(theta1[t]) + L2*sin(theta2[t])], color="yellow")
    # linkage 3 in CAD
    ax.plot([L2*cos(theta2[t]), L2*cos(theta2[t]) - L3*sin(theta1[t])],
            [L2*sin(theta2[t]), L2*sin(theta2[t]) + L3*cos(theta1[t])], color="green")

    # Set axes scales and draw grid
    ax.set_aspect('equal')
    plt.grid()


def IOtoThetaVals():
    # Choose whether to load data from existing file or use IO to create a new drawing
    print("do you want to run from existing file (yes/no)?")
    if str(input()) == "yes":
        print("input file path: ")
        file = str(input())
        # Get the stored dimensions and piece-wise function count from the file
        dims = np.loadtxt(file, max_rows=6, dtype=float)
        L1 = dims[0]
        L2 = dims[1]
        L3 = dims[2]
        Gx = dims[3]
        Gy = dims[4]
        pieces = dims[5]
        datastart = int(len(dims) + pieces)
        # Read the file columns into theta1 and theta2 as numpy arrays of floats
        theta1 = np.loadtxt(file,
                            usecols=0, skiprows=datastart, dtype=float)
        theta2 = np.loadtxt(file,
                            usecols=1, skiprows=datastart, dtype=float)
    else:
        print("input new file name: ")
        file = open(str(input()), "w")

        # Store dimensions using helper function
        dims = StoreDims(file)
        L1 = dims[0]
        L2 = dims[1]
        L3 = dims[2]
        Gx = dims[3]
        Gy = dims[4]
        # print(L1)

        # Calculate and store theta values using helper function
        theta1, theta2 = CalcThetaVals(file, L1, L2, L3)
        file.close()
    return L1, L2, L3, Gx, Gy, theta1, theta2


def main():
    # Get IO input to get dimensions and theta values
    L1, L2, L3, Gx, Gy, theta1, theta2 = IOtoThetaVals()
    # Pass dimensions and theta values into helper function to get radii and angle values
    a, r_bottom, r_top, c_bottom, c_top = CalcRadiiVals(
        L1, Gx, Gy, theta1, theta2)

    # Create and show animation using Matplotlib's Funcanimation function, passing in supplementary args
    fig, ax = plt.subplots()
    anim = FuncAnimation(fig, animate, fargs=(
        ax, L1, L2, L3, Gx, Gy, theta1, theta2, a, r_bottom, r_top, c_bottom, c_top), frames=360, interval=100, repeat=true)
    plt.show()

    # Offer to store animation as a gif
    print("Do you want to save the animation as a .gif? (yes/no)")
    if str(input()) == "yes":
        print("enter file name for animation, including .gif extension")
        name = str(input())
        anim.save(name)


main()
