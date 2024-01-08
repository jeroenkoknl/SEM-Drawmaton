import numpy as np
import sympy as sp
from sympy import *
from sympy.plotting import plot_parametric
from sympy.parsing.sympy_parser import parse_expr
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import pandas as pd

# a is an independent parameterized variable on [0,2*pi]
a = np.linspace(0, 2*np.pi, 100, dtype=float)


def StoreDims(file):
    print("use default dimensions?")
    if str(input()) == "yes":
        L1 = 7.6
        L2 = 15.8
        Gx = -15
        Gy = 7.6
    else:
        print("input L1 value: ")
        L1 = float(input())
        print("input L2 value: ")
        L2 = float(input())
        print("input Gx value: ")
        Gx = float(input())
        print("input Gy value: ")
        Gy = float(input())
    file.write(str(L1) + "\n")
    file.write(str(L2) + "\n")
    file.write(str(Gx) + "\n")
    file.write(str(Gy) + "\n")
    return np.array([L1, L2, Gx, Gy])


def StoreTargetXY(file):
    print("use default pen path (arabesque tile)?")
    if str(input()) == "yes":
        targetxfunc = 8*sin(4*t) + 18
        targetyfunc = 8*cos(3*t) + 18
        file.write(str(targetxfunc) + "\n")
        file.write(str(targetyfunc) + "\n")
    else:
        print("input x(t), the function for the pen path's x position: ")
        xinput = str(input())
        file.write(xinput + "\n")
        targetxfunc = parse_expr(xinput)
        print("input y(t), the function for the pen path's y position: ")
        yinput = str(input())
        file.write(yinput + "\n")
        targetyfunc = parse_expr(yinput)
    return np.array([targetxfunc, targetyfunc])


def CalcThetaVals(file, L1, L2, targetxfunc, targetyfunc):
    rawname = "Raw" + file.name
    rawfile = open(rawname, "w")
    CalcRawThetaVals(rawfile, L1, L2, targetxfunc, targetyfunc)
    return CleanThetaVals(rawfile.name, file)


def CalcRawThetaVals(rawfile, L1, L2, targetxfunc, targetyfunc):
    delim = " "
    L3 = L1 + L2
    th1, th2 = symbols('th1, th2', real=True)
    for i in range(len(a)):
        aval = a[i]
        eqx = Eq(L2*cos(th2) - L3*sin(th1), targetxfunc.subs(t, aval))
        eqy = Eq(L2*sin(th2) + L3*cos(th1), targetyfunc.subs(t, aval))
        soln = solve((eqx, eqy), (th1, th2))

        rawfile.write(str(soln[0][0]) + delim + str(soln[0][1]) +
                      delim + str(soln[1][0]) + delim + str(soln[1][1]) + "\n")
    print("finished raw vals")
    rawfile.close()


def CleanThetaVals(rawfile, file):
    rS0theta1 = np.loadtxt(rawfile, usecols=0, dtype=float)
    rS0theta2 = np.loadtxt(rawfile, usecols=1, dtype=float)
    rS1theta1 = np.loadtxt(rawfile, usecols=2, dtype=float)
    rS1theta2 = np.loadtxt(rawfile, usecols=3, dtype=float)

    theta1 = np.zeros(len(rS0theta1))
    theta2 = np.zeros(len(rS0theta2))
    for i in range(len(theta1)):
        if (i == 0 or np.abs(rS1theta1[i] - theta1[i-1]) <= 0.75):
            theta1[i] = rS1theta1[i]
            theta2[i] = rS1theta2[i]
        else:
            theta1[i] = rS0theta1[i]
            theta2[i] = rS0theta2[i]
        file.write(str(theta1[i]) + " " + str(theta2[i]) + "\n")
    print("finished clean")
    file.close()
    return np.array([theta1, theta2])


print("do you want to run from existing file?")
if str(input()) == "yes":
    print("input file path: ")
    file = str(input())
    dims = np.loadtxt(file, max_rows=4, dtype=float)
    L1 = dims[0]
    L2 = dims[1]
    L3 = L1 + L2
    Gx = dims[2]
    Gy = dims[3]
    theta1 = np.loadtxt(file,
                        usecols=0, skiprows=6, dtype=float)
    theta2 = np.loadtxt(file,
                        usecols=1, skiprows=6, dtype=float)
    # print(theta1)
    # print("\n", theta2)
else:
    t = symbols('t')
    print("input new file name: ")
    file = open(str(input()), "w")
    # nf = "Raw" + file.name
    # print(nf)

    dims = StoreDims(file)
    L1 = dims[0]
    L2 = dims[1]
    L3 = L1 + L2
    Gx = dims[2]
    Gy = dims[3]
    # print(L1)

    targetfuncs = StoreTargetXY(file)
    targetxfunc = targetfuncs[0]
    targetyfunc = targetfuncs[1]

    print("calculating theta values...")
    thetas = CalcThetaVals(file, L1, L2, targetxfunc, targetyfunc)
    theta1 = thetas[0]
    theta2 = thetas[1]
    # L3 = L1 + L2
    # th1, th2 = symbols('th1, th2', real=True)
    # theta1 = np.zeros(len(a))
    # theta2 = np.zeros(len(a))
    # for i in range(len(a)):
    #     if (i >= 12):
    #         break
    #     aval = a[i]
    #     eqx = Eq(L2*cos(th2) - L3*sin(th1), targetxfunc.subs(t, aval))
    #     eqy = Eq(L2*sin(th2) + L3*cos(th1), targetyfunc.subs(t, aval))
    #     soln = solve((eqx, eqy), (th1, th2))

    #     # solve[1] is correct for hypotrochoid
    #     if (i == 0 or np.abs(soln[1][0] - theta1[i-1]) < 1):
    #         soln = soln[1]
    #     else:
    #         soln = soln[0]
    #     file.write(str(soln[0]) + " " + str(soln[1]) + "\n")
    #     theta1[i] = soln[0]
    #     theta2[i] = soln[1]
    # print("finished")
    # file.close()

# print(theta1)
# print("\n", theta2)
fig, ax = plt.subplots()

# b is the angle relating the center of the rotors to the corresponding linkage point of contact
b_top = np.arctan2(-Gy + L1*np.cos(theta1), -Gx - L1*np.sin(theta1))
c_top = -a + b_top
r_top = ((-Gx - L1*np.sin(theta1))**2 + (-Gy + L1*np.cos(theta1))**2)**0.5

b_bottom = np.arctan2(-Gy + L1*np.cos(theta2), -Gx - L1*np.sin(theta2))
c_bottom = -a + b_bottom
r_bottom = ((-Gx - L1*np.sin(theta2))**2 + (-Gy + L1*np.cos(theta2))**2)**0.5


def animate(t):
    # print(t)
    i = t
    t = a[i % len(a)]

    ax.clear()
    ax.plot(0, 0, "o", color="black")
    # objects & paths:
    # pen path
    ax.plot(L2*np.cos(theta2) - L3*np.sin(theta1),
            L2*np.sin(theta2) + L3*np.cos(theta1), color="red")
    # top rotor
    ax.plot(r_top*np.cos(c_top)*np.cos(t) - r_top*np.sin(c_top)*np.sin(t) + Gx,
            r_top*np.cos(c_top)*np.sin(t) + r_top*np.sin(c_top)*np.cos(t) + Gy, color="green")
    # bottom rotor
    ax.plot(r_bottom*np.cos(c_bottom)*np.cos(t) - r_bottom*np.sin(c_bottom)*np.sin(t) + Gx,
            r_bottom*np.cos(c_bottom)*np.sin(t) + r_bottom*np.sin(c_bottom)*np.cos(t) + Gy, color="blue")

    # moving points
    t = i % len(a)

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

    # ax.set_xlim([-35, 30])
    # ax.set_ylim([-12, 30])
    ax.set_aspect('equal')
    plt.grid()


anim = FuncAnimation(fig, animate,
                     frames=360, interval=100, repeat=true)
plt.show()
