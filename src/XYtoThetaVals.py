import numpy as np
from sympy import *
from scipy.optimize import fsolve
import time


def CalcRawThetaValsXY(L1, L2, L3, xcoords, ycoords):
    start_time = time.time()  # Record the start time

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
        
    end_time = time.time()  # Record the end time
    elapsed_time = end_time - start_time
    print(f"Sympy Total runtime: {elapsed_time} seconds")
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

def CalcRawThetaValsXYfsolve(L1, L2, L3, xcoords, ycoords):
    start_time = time.time()  # Record the start time

    # Vectorized function for fsolve
    def func(thetas, i):
        return [L2*np.cos(thetas[1]) - L3*np.sin(thetas[0]) - xcoords[i], L2*np.sin(thetas[1]) + L3*np.cos(thetas[0]) - ycoords[i]]
    
    def jac(thetas, i):
        return [[-L3*np.cos(thetas[0]), -L2*np.sin(thetas[1])], [-L3*np.sin(thetas[0]), L2*np.cos(thetas[1])]]
    
    raw_solns = np.zeros((len(xcoords), 4))
    initial_guess = [0,0]
    for i in range(len(xcoords)):
        # print(initial_guess)
        # Solve using fsolve
        soln = fsolve(func, initial_guess, args=(i))
        # print(f"{i} solving for ({xcoords[i]},{ycoords[i]})", soln)
        # Store the solutions
        raw_solns[i, :] = soln[0], soln[1], soln[0], soln[1]
        initial_guess = [soln[0], soln[1]]
    
    end_time = time.time()  # Record the end time
    elapsed_time = end_time - start_time
    print(f"FSolve Total runtime: {elapsed_time} seconds")
    
    return raw_solns

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