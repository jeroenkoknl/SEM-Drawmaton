import SimulateDrawmaton as sim
import FileIOFunctions as fiof
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np

def PlotXY(simulation_filename):
    _, coords, _, _ = fiof.ReadDrawmatonSimulation(simulation_filename)
    fig, ax = plt.subplots()
    ax.plot(coords[:, 0], coords[:, 1])
    ax.set_aspect('equal')
    plt.show()
    
def CalculateRotorToBaseGap(simulation_filename):
    dims, _, _, radiiVals = fiof.ReadDrawmatonSimulation(simulation_filename)
    max_r = np.max(radiiVals)
    _, _, _, Gx, Gy = dims
    return (Gx**2 + Gy**2)**0.5, max_r, (Gx**2 + Gy**2)**0.5 - max_r

def ExportAnimation(simulation_filename, animation_filename):
    anim = sim.AnimateDrawmaton(simulation_filename, show=False)
    anim.save(filename=animation_filename)
    
def ExportRotorsSVG(simulation_filename, bottom_rotor_filename, top_rotor_filename):
        _, _, _, radiiVals = fiof.ReadDrawmatonSimulation(simulation_filename)
        r_bottom = radiiVals[:, 1]
        r_top = radiiVals[:, 2]
        c_bottom = radiiVals[:, 3]
        c_top = radiiVals[:, 4]
        
        matplotlib.use("svg")
        fig, ax = plt.subplots()
        
        ax.plot(r_bottom*np.cos(c_bottom), r_bottom*np.sin(c_bottom))
        ax.set_aspect('equal')
        ax.axis(False)
        refcirc = plt.Circle((0, 0), radius=0.5, color='blue', fill=False)
        orientcirc = plt.Circle((0, 1.5), radius=0.25,
                                color='green', fill=False)
        plt.gca().add_artist(refcirc)
        plt.gca().add_artist(orientcirc)
        plt.savefig(bottom_rotor_filename, format='svg')

        ax.clear()
        ax.plot(r_top*np.cos(c_top), r_top*np.sin(c_top))
        ax.set_aspect('equal')
        ax.axis(False)
        plt.gca().add_artist(refcirc)
        plt.gca().add_artist(orientcirc)
        plt.savefig(top_rotor_filename, format='svg')