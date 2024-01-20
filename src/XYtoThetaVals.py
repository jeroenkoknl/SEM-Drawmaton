import numpy as np
from sympy import *
from scipy.optimize import fsolve
import time

def CalcThetaVals(L1, L2, L3, xcoords, ycoords):
    start_time = time.time()  # Record the start time

    # Vectorized function for fsolve
    def func(thetas, i):
        return [L2*np.cos(thetas[1]) - L3*np.sin(thetas[0]) - xcoords[i], L2*np.sin(thetas[1]) + L3*np.cos(thetas[0]) - ycoords[i]]
    
    def jac(thetas, i):
        return [[-L3*np.cos(thetas[0]), -L2*np.sin(thetas[1])], [-L3*np.sin(thetas[0]), L2*np.cos(thetas[1])]]
    
    solns = np.zeros((len(xcoords), 2))
    initial_guess = [0,0]
    for i in range(len(xcoords)):
        # print(initial_guess)
        # Solve using fsolve
        soln = fsolve(func, initial_guess, args=(i))
        # print(f"{i} solving for ({xcoords[i]},{ycoords[i]})", soln)
        # Store the solutions
        solns[i, :] = soln[0], soln[1]
        initial_guess = [soln[0], soln[1]]
    
    end_time = time.time()  # Record the end time
    elapsed_time = end_time - start_time
    print(f"FSolve Total runtime: {elapsed_time} seconds")
    
    return solns[:,0], solns[:,1]

# def CalcRawThetaValsXY(L1, L2, L3, xcoords, ycoords):
#     start_time = time.time()  # Record the start time

#     # Turns the target pen path's X,Y coordinates into theta1 and theta2 values
#     # based on the linkage dimensions, using sympy's solver
#     th1, th2 = symbols('th1, th2', real=True)
#     # Iterate through all x,y coordinates, calculate theta values
#     raw_solns = np.zeros((len(xcoords), 4))
#     for i in range(len(xcoords)):
#         eqx = Eq(L2*cos(th2) - L3*sin(th1), xcoords[i])
#         eqy = Eq(L2*sin(th2) + L3*cos(th1), ycoords[i])
#         soln = solve((eqx, eqy), (th1, th2))
#         print(f"{i} solving for ({xcoords[i]},{ycoords[i]})", soln)
#         raw_solns[i:,] = np.array([soln[0][0], soln[0][1], soln[1][0], soln[1][1]])
#         # rawfile.write(str(soln[0][0]) + delim + str(soln[0][1]) +
#         #               delim + str(soln[1][0]) + delim + str(soln[1][1]) + "\n")
        
#     end_time = time.time()  # Record the end time
#     elapsed_time = end_time - start_time
#     print(f"Sympy Total runtime: {elapsed_time} seconds")
#     return raw_solns

# def CleanThetaVals(raw_vals, diff_cutoff=0.6):
#     # Clean the theta values from the raw solution by choosing the correct values from the two possible solution pairs
#     print("Cleaning theta values")
    
#     # The final solution theta1 and theta2 arrays
#     theta1 = np.zeros(len(raw_vals))
#     theta2 = np.zeros(len(raw_vals))
#     for i in range(len(theta1)):
#         # Choose the 1st solution pair by default, or if the previous theta values were within diff_cutoff radians margin
#         # from the current 1st solution theta values
#         if (i == 0 or (np.abs(raw_vals[i,2] - theta1[i-1]) <= diff_cutoff and np.abs(raw_vals[i,3] - theta2[i-1]) <= diff_cutoff)):
#             theta1[i] = raw_vals[i,2]
#             theta2[i] = raw_vals[i,3]
#         else:
#             # Choose the 0th solution pair theta values instead
#             theta1[i] = raw_vals[i,0]
#             theta2[i] = raw_vals[i,1]
#     print("finished clean")
#     # print(theta1, theta2)
#     return theta1, theta2

