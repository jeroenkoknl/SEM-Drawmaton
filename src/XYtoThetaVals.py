import numpy as np
from sympy import *

def CalcRawThetaValsXY(L1, L2, L3, xcoords, ycoords):
    # Turns the target pen path's X,Y coordinates into theta1 and theta2 values
    # based on the linkage dimensions, using sympy's solver
    th1, th2 = symbols('th1, th2', real=True)
    # Iterate through all x,y coordinates, calculate theta values
    raw_solns = np.zeros((len(xcoords), 4))
    for i in range(len(xcoords)):
        eqx = Eq(L2*cos(th2) - L3*sin(th1), xcoords[i])
        eqy = Eq(L2*sin(th2) + L3*cos(th1), ycoords[i])
        soln = solve((eqx, eqy), (th1, th2))
        print(f"{i} solving for ({xcoords[i]},{ycoords[i]})", soln)
        raw_solns[i:,] = np.array([soln[0][0], soln[0][1], soln[1][0], soln[1][1]])
        # rawfile.write(str(soln[0][0]) + delim + str(soln[0][1]) +
        #               delim + str(soln[1][0]) + delim + str(soln[1][1]) + "\n")
    return raw_solns

def CleanThetaVals(raw_vals, diff_cutoff=0.6):
    # Clean the theta values from the raw solution by choosing the correct values from the two possible solution pairs
    print("Cleaning theta values")
    
    # The final solution theta1 and theta2 arrays
    theta1 = np.zeros(len(raw_vals))
    theta2 = np.zeros(len(raw_vals))
    for i in range(len(theta1)):
        # Choose the 1st solution pair by default, or if the previous theta values were within diff_cutoff radians margin
        # from the current 1st solution theta values
        if (i == 0 or (np.abs(raw_vals[i,2] - theta1[i-1]) <= diff_cutoff and np.abs(raw_vals[i,3] - theta2[i-1]) <= diff_cutoff)):
            theta1[i] = raw_vals[i,2]
            theta2[i] = raw_vals[i,3]
        else:
            # Choose the 0th solution pair theta values instead
            theta1[i] = raw_vals[i,0]
            theta2[i] = raw_vals[i,1]
    print("finished clean")
    # print(theta1, theta2)
    return theta1, theta2

# def CalcRawThetaValsXY(rawfile, L1, L2, L3, xcoords, ycoords):
#     # Turns the target pen path's X,Y coordinates into theta1 and theta2 values
#     # based on the linkage dimensions, using sympy's solver
#     delim = " "
#     th1, th2 = symbols('th1, th2', real=True)
#     # Iterate through all x,y coordinates, calculate, and store both sympy solutions in a raw data file
#     for i in range(len(xcoords)):
#         eqx = Eq(L2*cos(th2) - L3*sin(th1), xcoords[i])
#         eqy = Eq(L2*sin(th2) + L3*cos(th1), ycoords[i])
#         soln = solve((eqx, eqy), (th1, th2))
#         print(f"{i} solving for ({xcoords[i]},{ycoords[i]})", soln)
#         rawfile.write(str(soln[0][0]) + delim + str(soln[0][1]) +
#                       delim + str(soln[1][0]) + delim + str(soln[1][1]) + "\n")
#     rawfile.close()


# def CleanThetaVals(rawfile, file):
#     # Clean the theta values from the rawfile by choosing the correct values from the two possible solution pairs
#     print("Cleaning theta values")
#     # The theta 1 and 2 values from the 0th and 1st solution pairs that sympy solve generates
#     rS0theta1 = np.loadtxt(rawfile, usecols=0, dtype=float)
#     rS0theta2 = np.loadtxt(rawfile, usecols=1, dtype=float)
#     rS1theta1 = np.loadtxt(rawfile, usecols=2, dtype=float)
#     rS1theta2 = np.loadtxt(rawfile, usecols=3, dtype=float)

#     # The final solution theta1 and theta2 arrays
#     theta1 = np.zeros(len(rS0theta1))
#     theta2 = np.zeros(len(rS0theta2))
#     for i in range(len(theta1)):
#         # Choose the 1st solution pair by default, or if the previous theta values were within a 0.6 radian margin
#         # from the current 1st solution theta values
#         if (i == 0 or (np.abs(rS1theta1[i] - theta1[i-1]) <= 0.6 and np.abs(rS1theta2[i] - theta2[i-1]) <= 0.6)):
#             theta1[i] = rS1theta1[i]
#             theta2[i] = rS1theta2[i]
#         else:
#             # Choose the 0th solution pair theta values instead
#             theta1[i] = rS0theta1[i]
#             theta2[i] = rS0theta2[i]
#         file.write(str(theta1[i]) + " " + str(theta2[i]) + "\n")
#     print("finished clean")
#     file.close()
#     return theta1, theta2