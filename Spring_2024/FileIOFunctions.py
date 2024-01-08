import numpy as np
from sympy import *
from sympy.parsing.sympy_parser import parse_expr

# WRITE FUNCTIONS
def SetUpFile(simulation_filename):
    simulation_file = open(simulation_filename, "w")
    simulation_file.write("$SimulationHeader\n")
    simulation_file.write("$EndSimulationHeader\n")
    simulation_file.close()
        
def StoreDims(simulation_filename, L1, L2, L3, Gx, Gy):
    simulation_file = open(simulation_filename, "a")
    simulation_file.write("$Dimensions\n")
    # simulation_file.write("L1, L2, L3, Gx, Gy:\n")
    simulation_file.write(str(L1) + "\n")
    simulation_file.write(str(L2) + "\n")
    simulation_file.write(str(L3) + "\n")
    simulation_file.write(str(Gx) + "\n")
    simulation_file.write(str(Gy) + "\n")
    simulation_file.write("$EndDimensions\n")
    simulation_file.close()
    
def StoreImagePath(simulation_filename, drawing_src_filename):
    simulation_file = open(simulation_filename, "a")
    simulation_file.write("$DrawingSource\n")
    simulation_file.write(drawing_src_filename + "\n")
    simulation_file.write("$EndDrawingSource\n")
    simulation_file.close()

def StoreParametricEQs(simulation_filename, targetxfuncs, targetyfuncs, starts, ends):
    simulation_file = open(simulation_filename, "a")
    simulation_file.write("$ParametricEquations\n")
    
    piececount = len(targetxfuncs)
    for i in range(piececount):
        xinput = targetxfuncs[i]
        yinput = targetyfuncs[i]
        startinput = starts[i]
        endinput = ends[i]
        # Write x,y functions and start,end pairs to file
        simulation_file.write(str(xinput) + ", " + str(yinput) + ", " +
                   str(startinput) + ", " + str(endinput) + "\n")
    simulation_file.write("$EndParametricEquations\n")
    simulation_file.close()
    
def StoreDataCount(simulation_filename, count):
    simulation_file = open(simulation_filename, "a")
    simulation_file.write("$DataCount\n")    
    simulation_file.write(str(count) + "\n")
    simulation_file.write("$EndDataCount\n")
    simulation_file.close()
    
def StoreXYCoords(simulation_filename, xcoords, ycoords):
    simulation_file = open(simulation_filename, "a")
    simulation_file.write("$XYCoordinates\n")

    for i in range(len(xcoords)):
        simulation_file.write(str(xcoords[i]) + " " + str(ycoords[i]) + "\n")
    
    simulation_file.write("$EndXYCoordinates\n")
    simulation_file.close()
    
def StoreThetaVals(simulation_filename, theta1, theta2):
    simulation_file = open(simulation_filename, "a")
    simulation_file.write("$ThetaValues\n")
    
    for i in range(len(theta1)):
        simulation_file.write(str(theta1[i]) + " " + str(theta2[i]) + "\n")
    simulation_file.write("$EndThetaValues\n")
    simulation_file.close()

def StoreRawThetaVals(simulation_filename, raw_vals):
    simulation_file = open(simulation_filename, "a")
    simulation_file.write("$RawThetaValues\n")
    
    for i in range(len(raw_vals)):
        simulation_file.write(str(raw_vals[i,0]) + " " + str(raw_vals[i,1]) + " " + str(raw_vals[i,2]) + " " + str(raw_vals[i,3]) + "\n")
    simulation_file.write("$EndRawThetaValues\n")
    simulation_file.close()

def StoreRadiiVals(simulation_filename, a, r_bottom, r_top, c_bottom, c_top):
    simulation_file = open(simulation_filename, "a")
    simulation_file.write("$RotorRadiiValues\n")
    for i in range(len(a)):
        simulation_file.write(str(a[i]) + " " + str(r_bottom[i]) + " " + str(r_top[i]) + " " + str(c_bottom[i]) + " " + str(c_top[i]) + "\n")
    simulation_file.write("$EndRotorRadiiValues\n")
    simulation_file.close()        
        
    
# READ FUNCTIONS
def ReadDrawmatonSimulation(simulation_filename):
    simulation_file = open(simulation_filename)
    simData = simulation_file.readlines()
    
    # Dimensions
    dimsStartInd = simData.index("$Dimensions\n")
    dimsEndInd = simData.index("$EndDimensions\n")
    # print(dimsStartInd, dimsEndInd)
    dims = np.zeros(dimsEndInd - dimsStartInd  - 1)
    
    for i in range(dimsStartInd + 1, dimsEndInd):
        # print(simData[i].split()[0])
        dims[i - dimsStartInd - 1] = float(simData[i].split()[0])
    # print(dims)
    
    # DataCount
    countStartInd = simData.index("$DataCount\n")
    dataCount = int(simData[countStartInd + 1].split()[0])
    
    # XYCoordinates
    coordsStartInd = simData.index("$XYCoordinates\n")
    coordsEndInd = coordsStartInd + dataCount + 1
    coords = np.zeros((dataCount, 2))
    
    for i in range(coordsStartInd + 1, coordsEndInd):
        # print(simData[i].split())
        coords[i - coordsStartInd - 1:,] = np.array([float(simData[i].split()[0]), float(simData[i].split()[1])])        
    # print(coords)

    # Theta Values
    thetasStartInd = simData.index("$ThetaValues\n")
    thetasEndInd = thetasStartInd + dataCount + 1
    thetaVals = np.zeros((dataCount, 2))
    
    for i in range(thetasStartInd + 1, thetasEndInd):
        thetaVals[i - thetasStartInd - 1:,] = np.array([float(simData[i].split()[0]), float(simData[i].split()[1])])        
    # print(thetaVals)
    
    # Rotor Radii Values 
    # a, r_bottom, r_top, c_bottom, c_top
    radiiStartInd = simData.index("$RotorRadiiValues\n")
    radiiEndInd = radiiStartInd + dataCount + 1
    radiiVals = np.zeros((dataCount, 5))
    
    for i in range(radiiStartInd + 1, radiiEndInd):
        radiiVals[i - radiiStartInd - 1:,] = np.array([float(simData[i].split()[0]), float(simData[i].split()[1]), float(simData[i].split()[2]), float(simData[i].split()[3]), float(simData[i].split()[4])])
    # print(radiiVals)
    
    simulation_file.close()
    return dims, coords, thetaVals, radiiVals
    

def ReadStoredParametricEQs(parametric_eqs_filename):
    parametric_eqs_file = open(parametric_eqs_filename, "r")
    xfuncs = np.loadtxt(parametric_eqs_file, delimiter=",", usecols=0, dtype=str, ndmin=1)
    parametric_eqs_file.close()
    
    parametric_eqs_file = open(parametric_eqs_filename, "r")
    yfuncs = np.loadtxt(parametric_eqs_file, delimiter=",", usecols=1, dtype=str, ndmin=1)
    parametric_eqs_file.close()
    
    parametric_eqs_file = open(parametric_eqs_filename, "r")
    starts = np.loadtxt(parametric_eqs_file, delimiter=",",
                        usecols=2, dtype=float, ndmin=1)
    parametric_eqs_file.close()
    
    parametric_eqs_file = open(parametric_eqs_filename, "r")
    ends = np.loadtxt(parametric_eqs_file, delimiter=",", usecols=3, dtype=float, ndmin=1)
    parametric_eqs_file.close()
    
    # print(xfuncs, yfuncs, starts, ends)
    targetxfuncs = []
    targetyfuncs = []
    # Store the x,y function pairs and the domain start,end pairs, and using sympy's parse_expr to convert strings into sympy equations
    for i in range(len(xfuncs)):
        targetxfuncs.append(parse_expr(xfuncs[i]))
        targetyfuncs.append(parse_expr(yfuncs[i]))
    
    return targetxfuncs, targetyfuncs, starts, ends

def ReadStoredXYCoords(xycoords_filename):
    xycoords_file = open(xycoords_filename, "r")
    xcoords = np.loadtxt(xycoords_file, delimiter=" ", usecols=0, dtype=float)
    xycoords_file.close()
    xycoords_file = open(xycoords_filename, "r")
    ycoords = np.loadtxt(xycoords_file, delimiter=" ", usecols=1, dtype=float)
    xycoords_file.close()
    return xcoords, ycoords
    