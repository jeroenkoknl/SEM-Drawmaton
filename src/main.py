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
# L1 = 4.1
# L2 = 16.2
# L3 = 24.0
# Gx = -4.0
# Gy = 15.0

# # DIMS 6
L1 = 5.8
L2 = 16.6
L3 = 24.3
Gx = -4.5
Gy = 16.0

# hello
dims = np.array([L1, L2, L3, Gx, Gy])
drawing_src_filename = "./InputImages/TulipTest upside down.jpg"
drawing_src_type = "image"
simulation_filename = "./SimulationFiles/Dims6/Dims6TulipUpsideDown.txt"
# save_animation_filename = "./Animations/Dims6Arabesque.gif"
# sim.CreateDrawmatonSimulation(dims, drawing_src_filename, drawing_src_type, simulation_filename)
# print(util.CalculateRotorToBaseGap(simulation_filename))
# print(NDimArray(util.CalcMinSpace(simulation_filename)))
# sim.AnimateDrawmaton(simulation_filename)
# util.ExportAnimation(simulation_filename, save_animation_filename)
# util.CalcRadiusOfCurvatue(simulation_filename)
util.ExportRotorsSVG(simulation_filename, "./Rotors/Dims6TulipUpsideDownBottom.svg", "./Rotors/Dims6TulipUpsideDownTop.svg")
# util.PlotXY("./SimulationFiles/TulipDims5Sim.txt")
# util.CompareSimulations('./SimulationFiles/Dims5/ArabesqueFsolveDims5.txt', './SimulationFiles/Dims6/Dims6Arabesque.txt')