import SimulateDrawmaton as sim
import FileIOFunctions as fiof
import utilities as util
import numpy as np
from sympy import *
import matplotlib
import matplotlib.pyplot as plt

# DIMS 1
# L1 = 7.6
# L2 = 15.8
# L3 = 23.4
# Gx = -15
# Gy = 7.6

# DIMS 5
L1 = 4.1
L2 = 16.2
L3 = 24.0
Gx = -4.0
Gy = 15.0

# # DIMS 6
# L1 = 5.7
# L2 = 16.6
# L3 = 24.5
# Gx = -4.5
# Gy = 16.0

dims = np.array([L1, L2, L3, Gx, Gy])
drawing_src_filename = "./InputImages/blackcat.jpg"
drawing_src_type = "image"
simulation_filename = "./SimulationFiles/ArabesqueFSolveDims5.txt"
save_animation_filename = "./Animations/BlackCatFSolveDims5.gif"
# sim.CreateDrawmatonSimulation(dims, drawing_src_filename, drawing_src_type, simulation_filename)
sim.AnimateDrawmaton(simulation_filename)
# print(util.CalculateRotorToBaseGap(ysimulation_filename))
# util.ExportAnimation(simulation_filename, save_animation_filename)
# util.ExportRotorsSVG(simulation_filename, "./Rotors/WomanProfileDims5Bottom.svg", "./Rotors/WomanProfileDims5Top.svg")
# util.PlotXY("./SimulationFiles/TulipDims5Sim.txt")