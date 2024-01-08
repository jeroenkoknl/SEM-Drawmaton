from random import randint
import numpy as np
from sympy import *
from sympy.plotting import plot_parametric
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


t = symbols('t', real=True)
r = 2*cos(5*t) + 13

frame = plot(0, (t, 0, 0))


def draw_frame(a):
    a = np.radians(a)
    x = r*cos(t)*cos(a) - r*sin(t)*sin(a)
    y = r*cos(t)*sin(a) + r*sin(t)*cos(a)

    frame = plot_parametric(
        (x, y), (t, 0, 2*np.pi), xlim=(-18, 18), ylim=(-18, 18))
    # frame.show()


def animate_frames(frames, interval):
    i = 0
    while i <= frames:
        draw_frame(i)
        i += interval


animate_frames(20, 2*np.pi/20)

# def parametric_animate(a):
#     a = np.radians(a)
#     x = r*cos(a)*cos(a) - r*sin(a)*sin(a)
#     y = r*cos(a)*sin(a) + r*sin(a)*cos(a)

#     ax.clear()
#     ax = plot_parametric((x, y), (t, 0, 2*np.pi))
#     ax.set_xlim([-16, 16])
#     ax.set_ylim([-16, 16])


# t = symbols('t', real=True)
# r = 2*cos(5*t) + 13

# fig, ax = plt.subplots()


# def parametric_animate(a):
#     a = np.radians(a)
#     x = r*cos(a)*cos(a) - r*sin(a)*sin(a)
#     y = r*cos(a)*sin(a) + r*sin(a)*cos(a)

#     ax.clear()
#     ax = plot_parametric((x, y), (t, 0, 2*np.pi))
#     ax.set_xlim([-16, 16])
#     ax.set_ylim([-16, 16])


# para_ani = FuncAnimation(fig, parametric_animate,
#                          frames=40, interval=250, repeat=False)
# plt.show()
# para_ani.save("para_ani.gif")
# x = []
# y = []


# fig, ax = plt.subplots()


# def animate(i):
#     pt = randint(1, 9)
#     x.append(i)
#     y.append(pt)

#     ax.clear()
#     ax.plot(x, y)
#     ax.set_xlim([0, 20])
#     ax.set_ylim([0, 10])


# ani = FuncAnimation(fig, animate, frames=20, interval=500, repeat=False)
# plt.show()
# ani.save('testani.gif')
