import numpy as np
from sympy import *
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import ImageToXY as imgToXY
import SVGToXY as svgToXY
import ParametricToXY as parametricToXY
import FileIOFunctions as fiof
import XYtoThetaVals as xyToTheta
import ThetaValsToRadiiVals as thetasToRadii

def CreateDrawmatonSimulation(dims, drawing_src_filename, drawing_src_type, simulation_filename):
    print("\n=== Starting Drawmaton Simulation ===")
    print("1. Initializing simulation...")
    L1, L2, L3, Gx, Gy = dims
    fiof.SetUpFile(simulation_filename)
    fiof.StoreDims(simulation_filename, L1, L2, L3, Gx, Gy)
    if (drawing_src_type == "image"):
        # define the position on the xy grid that the image should occupy (min rectangle based on the geometry of the linkages)
        # temporary fixed values for Dims6
        targetx = 13.4
        targety = 27.1
        targetw = 16.3
        targeth = 16.3
        xcoords, ycoords = imgToXY.ImageToXY(L1, L2, L3, Gx, Gy, drawing_src_filename, targetx, targety, targetw, targeth) 
        fiof.StoreImagePath(simulation_filename, drawing_src_filename)
    elif (drawing_src_type == "parametric"):
        targetxfuncs, targetyfuncs, starts, ends = fiof.ReadStoredParametricEQs(drawing_src_filename)
        fiof.StoreParametricEQs(simulation_filename, targetxfuncs, targetyfuncs, starts, ends)
        datacount = 300 + 20*len(targetxfuncs)
        print(datacount)
        xcoords, ycoords = parametricToXY.ParametricToXY(targetxfuncs, targetyfuncs, starts, ends, datacount)
    elif (drawing_src_type == "svg"):
        targetx = 13.4
        targety = 27.1
        targetw = 16.3
        targeth = 16.3
        xcoords, ycoords = svgToXY.SVGtoXY(L1, L2, L3, Gx, Gy, drawing_src_filename, targetx, targety, targetw, targeth, 17)
        fiof.StoreImagePath(simulation_filename, drawing_src_filename)
    elif (drawing_src_type == 'coordinates'):
        xcoords, ycoords = fiof.ReadStoredXYCoords(drawing_src_filename)
    else:
        print("improper drawing src type flag")
        return
    print("3. Processing coordinates...")
    fiof.StoreDataCount(simulation_filename,len(xcoords))
    fiof.StoreXYCoords(simulation_filename, xcoords, ycoords)
    print("4. Calculating mechanical movements...")
    theta1, theta2 = xyToTheta.CalcThetaVals(L1, L2, L3, xcoords, ycoords)
    # rawS0theta1, rawS0theta2, rawS1theta1, rawS1theta2 = xyToTheta.CalcRawThetaValsXY()
    # raw_theta_vals = xyToTheta.CalcRawThetaValsXYfsolve(L1, L2, L3, xcoords, ycoords)
    # diff_cutoff = 0.6
    # theta1, theta2 = xyToTheta.CleanThetaVals(raw_theta_vals, diff_cutoff)
    # fiof.StoreRawThetaVals(simulation_filename, raw_theta_vals)
    fiof.StoreThetaVals(simulation_filename, theta1, theta2)
    a, r_bottom, r_top, c_bottom, c_top = thetasToRadii.CalcRadiiVals(L1, Gx, Gy, theta1, theta2)
    fiof.StoreRadiiVals(simulation_filename, a, r_bottom, r_top, c_bottom, c_top)
    
    
def AnimateDrawmaton(simulation_filename, frames=360, interval=20, repeat=True, show=True):
    # Create a new figure and axes
    fig = plt.figure(figsize=(10, 10))
    ax = plt.axes()
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
        ax.set_xlim(Gx - 1.25*np.sqrt(Gx**2 + Gy**2), 0.9*(L2 + L3))
        ax.set_ylim(-L2, 0.9*(L2 + L3))
        plt.grid()
    
    # if (fig != None):
    #     fig, ax = plt.subplots()
    dims, coords, thetaVals, radiiVals = fiof.ReadDrawmatonSimulation(simulation_filename)
    L1, L2, L3, Gx, Gy = dims
    theta1 = thetaVals[:, 0]
    theta2 = thetaVals[:, 1]
    a = radiiVals[:, 0]
    r_bottom = radiiVals[:, 1]
    r_top = radiiVals[:, 2]
    c_bottom = radiiVals[:, 3]
    c_top = radiiVals[:, 4]
    frames = len(a)
    interval = 1000//60
    print("   - Starting animation (this may take a few seconds)...")
    anim = FuncAnimation(fig, animate, fargs=(ax, L1, L2, L3, Gx, Gy, theta1, theta2, a, r_bottom, r_top, c_bottom, c_top), 
                        frames=frames, interval=20, repeat=repeat)
    if show:
        plt.show(block=True)
    return anim
        
    
    

