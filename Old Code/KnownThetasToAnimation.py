import numpy as np
import sympy as sp
from sympy import *
from sympy.plotting import plot_parametric
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# a is an independent parameterized variable on [0,2*pi]
a = np.linspace(0, 2*np.pi, 100, dtype=float)
L1 = 7.6
L2 = 15.8
Gx = -15
Gy = L1
theta1 = 3*a
theta2 = 4*a
# temporary radius of bottom rotor function
# r = 2*np.cos(5*a) + 13

# b = np.linspace(0, 1, 3, dtype=float)
# top_linkage = np.array([-b*L1*np.sin(theta1), b*L1*np.cos(theta1)])
# bottom_linkage = np.array([-b*L1*np.sin(theta2), b*L1*np.cos(theta2)])

# b is the angle relating the center of the rotors to the corresponding linkage point of contact
b_top = np.arctan2(-Gy + L1*np.cos(theta1), -Gx - L1*np.sin(theta1))
c_top = -a + b_top
r_top = ((-Gx - L1*np.sin(theta1))**2 + (-Gy + L1*np.cos(theta1))**2)**0.5

b_bottom = np.arctan2(-Gy + L1*np.cos(theta2), -Gx - L1*np.sin(theta2))
c_bottom = -a + b_bottom
r_bottom = ((-Gx - L1*np.sin(theta2))**2 + (-Gy + L1*np.cos(theta2))**2)**0.5

fig, ax = plt.subplots()
print(theta1)


def animate(t):
    i = t
    t = np.radians(t)
    theta3 = 3*t
    theta4 = 4*t

    ax.clear()
    ax.plot(0, 0, "o")
    # objects & paths:
    # pen path
    ax.plot(L2*np.cos(theta2) - (L1 + L2)*np.sin(theta1),
            L2*np.sin(theta2) + (L1 + L2)*np.cos(theta1))
    # top rotor
    ax.plot(r_top*np.cos(c_top)*np.cos(t) - r_top*np.sin(c_top)*np.sin(t) + Gx,
            r_top*np.cos(c_top)*np.sin(t) + r_top*np.sin(c_top)*np.cos(t) + Gy)
    # bottom rotor
    ax.plot(r_bottom*np.cos(c_bottom)*np.cos(t) - r_bottom*np.sin(c_bottom)*np.sin(t) + Gx,
            r_bottom*np.cos(c_bottom)*np.sin(t) + r_bottom*np.sin(c_bottom)*np.cos(t) + Gy)

    # moving points
    # pen position
    ax.plot(L2*cos(theta4) - (L1 + L2)*sin(theta3), L2 *
            sin(theta4) + (L1 + L2)*cos(theta3), ".", color='red')
    # top linkage contact point
    ax.plot(-L1*sin(theta3), L1*cos(theta3), ".", color='green')
    # bottom linkage contact point
    ax.plot(-L1*sin(theta4), L1*cos(theta4), ".", color="blue")
    ax.set_xlim([-55, 55])
    ax.set_ylim([-55, 55])
    ax.set_aspect('equal')
    plt.grid()


anim = FuncAnimation(fig, animate,
                     frames=360, interval=16, repeat=true)
plt.show()
