import SimulateDrawmaton as sim
import FileIOFunctions as fiof
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Rectangle

def PlotXY(simulation_filename):
    _, coords, _, _ = fiof.ReadDrawmatonSimulation(simulation_filename)
    fig, ax = plt.subplots()
    ax.plot(coords[:, 0], coords[:, 1], color='red')
    # ax.scatter([0],[0], color='red')
    ax.set_aspect('equal')
    # ax.set_xlim(Gx - 1.25*np.sqrt(Gx**2 + Gy**2), 0.9*(L2 + L3))
    # ax.set_ylim(-L2, 0.9*(L2 + L3))
    # ax.axis(False)
    plt.show()
    
def CalculateRotorToBaseGap(simulation_filename):
    dims, _, _, radiiVals = fiof.ReadDrawmatonSimulation(simulation_filename)
    max_r = np.max(radiiVals[:, 1:3])
    _, _, _, Gx, Gy = dims
    return (Gx**2 + Gy**2)**0.5, max_r, (Gx**2 + Gy**2)**0.5 - max_r

def CalcRadiusOfCurvatue(simulation_filename):
    _, _, _, radiiVals = fiof.ReadDrawmatonSimulation(simulation_filename)
    a = radiiVals[:, 0]
    r_bottom = radiiVals[:, 1]
    r_top = radiiVals[:, 2]
    spacing = a[1] - a[0]
    dr_bottom = np.gradient(r_bottom, spacing)
    d2r_bottom = np.gradient(dr_bottom, spacing)
    curvature_bottom = (r_bottom**2 + dr_bottom**2)**1.5 / (r_bottom**2 + 2*dr_bottom**2 - r_bottom*d2r_bottom)
    fig, ax = plt.subplots()
    # ax.plot(a, r_bottom, color='red')
    # ax.plot(a, dr_bottom, color='green')
    # ax.plot(a, d2r_bottom, color='blue')
    ax.plot(r_bottom*np.cos(a), r_bottom*np.sin(a))
    ax.plot(a, curvature_bottom, color = 'orange')
    ax.plot(curvature_bottom*np.cos(a), curvature_bottom*np.sin(a))    
    ax.set_xlim(-35,35)
    ax.set_ylim(-35,35)
    ax.set_aspect('equal')
    plt.grid()
    plt.show()
    
def CompareSimulations(sim_file1, sim_file2):
    fig, ax = plt.subplots()
    anim1 = sim.AnimateDrawmaton(sim_file1, fig=fig, ax=ax)
    anim2 = sim.AnimateDrawmaton(sim_file2, fig=fig, ax=ax)
    plt.show()
    

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
        orientcirc1 = plt.Circle((0, 1.5), radius=0.25,
                                color='green', fill=False)
        orientcirc2 = plt.Circle((1.5, 0), radius=0.25,
                                color='green', fill=False)
        plt.gca().add_artist(refcirc)
        plt.gca().add_artist(orientcirc1)
        plt.gca().add_artist(orientcirc2)
        cutoutrect1 = Rectangle((-2.07/2, 1.465), 2.07, 1.07, color = 'red', fill=False)
        cutoutrect2 = Rectangle((-2.07/2, - 1.465 - 1.07), 2.07, 1.07, color = 'orange', fill=False)
        cutoutrect3 = Rectangle((1.465, -2.07/2), 1.07, 2.07, color = 'yellow', fill=False)
        cutoutrect4 = Rectangle((-1.465 - 1.07, -2.07/2), 1.07, 2.07, color = 'green', fill=False)
        
        ax.add_patch(cutoutrect1)
        ax.add_patch(cutoutrect2)
        ax.add_patch(cutoutrect3)
        ax.add_patch(cutoutrect4)


        plt.savefig(bottom_rotor_filename, format='svg')

        ax.clear()
        ax.plot(r_top*np.cos(c_top), r_top*np.sin(c_top))
        ax.set_aspect('equal')
        ax.axis(False)
        orientcirc1 = plt.Circle((0, 1.15), radius=0.25,
                                color='green', fill=False)
        orientcirc2 = plt.Circle((1.15, 0), radius=0.25,
                                color='green', fill=False)
        plt.gca().add_artist(refcirc)
        plt.gca().add_artist(orientcirc1)
        plt.gca().add_artist(orientcirc2)

        cutoutrect1 = Rectangle((-1.57/2, 1.715), 1.57, 0.57, color = 'red', fill=False)
        cutoutrect2 = Rectangle((-1.57/2, - 1.715 - .57), 1.57, 0.57, color = 'orange', fill=False)
        cutoutrect3 = Rectangle((1.715, -1.57/2), 0.57, 1.57, color = 'yellow', fill=False)
        cutoutrect4 = Rectangle((-1.715 - 0.57, -1.57/2), 0.57, 1.57, color = 'green', fill=False)
        
        ax.add_patch(cutoutrect1)
        ax.add_patch(cutoutrect2)
        ax.add_patch(cutoutrect3)
        ax.add_patch(cutoutrect4)

        plt.savefig(top_rotor_filename, format='svg')
        