import numpy as np
import sympy as sp
from sympy import *
from sympy.plotting import plot_parametric
from sympy.parsing.sympy_parser import parse_expr
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import pandas as pd

# a is an independent parameterized variable on [0,2*pi]
a = np.linspace(0, 2*np.pi, 125, dtype=float)


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
        targetxfuncs = [targetxfunc]
        targetyfuncs = [targetyfunc]
        starts = np.array([0])
        ends = np.array([2*np.pi])
        return [targetxfuncs, targetyfuncs, starts, ends]

    return StorePieceWiseXY(file)


def StorePieceWiseXY(file):
    targetxfuncs = []
    targetyfuncs = []
    print("read piece wise functions from file?")
    if (str(input()) == "yes"):
        return ReadStoredPieceWiseFuncs(file)
    print("how many pieces? (at most 20)")
    piececount = int(input())
    file.write(str(piececount) + "\n")
    starts = np.zeros(piececount)
    ends = np.zeros(piececount)
    if piececount > 1:
        print("Piece-wise mode: make sure that the starting time is 0 and ending time is 6.2831...(2*pi)")
    for i in range(piececount):
        print("input x" + str(i+1) +
              "(t), the function for the pen path's x position: ")
        xinput = str(input())
        print("input y" + str(i+1) +
              "(t), the function for the pen path's y position: ")
        yinput = str(input())
        if piececount == 1:
            startinput = 0
            endinput = 2*np.pi
        else:
            print("input the starting tval for this XY pair")
            startinput = float(input())
            print("input the ending tval for this XY pair")
            endinput = float(input())

        file.write(xinput + ", " + yinput + ", " +
                   str(startinput) + ", " + str(endinput) + "\n")
        targetxfuncs.append(parse_expr(xinput))
        targetyfuncs.append(parse_expr(yinput))
        starts[i] = startinput
        ends[i] = endinput
    return [targetxfuncs, targetyfuncs, starts, ends]


def ReadStoredPieceWiseFuncs(wfile):
    print("enter file name for stored piecewise functions:")
    rfile = str(input())
    xfuncs = np.loadtxt(rfile, delimiter=",", usecols=0, dtype=str)
    yfuncs = np.loadtxt(rfile, delimiter=",", usecols=1, dtype=str)
    starts = np.loadtxt(rfile, delimiter=",",
                        usecols=2, dtype=float)
    ends = np.loadtxt(rfile, delimiter=",", usecols=3, dtype=float)
    pieces = len(xfuncs)
    wfile.write(str(pieces) + "\n")
    targetxfuncs = []
    targetyfuncs = []
    for i in range(len(xfuncs)):
        wfile.write(xfuncs[i] + ", " + yfuncs[i] + ", " +
                    str(starts[i]) + ", " + str(ends[i]) + "\n")
        targetxfuncs.append(parse_expr(xfuncs[i]))
        targetyfuncs.append(parse_expr(yfuncs[i]))
    return [targetxfuncs, targetyfuncs, starts, ends]


def CalcThetaVals(file, L1, L2, targetxfuncs, targetyfuncs, starts, ends):
    print("Calculating theta values, please wait")
    rawname = "Raw" + file.name
    rawfile = open(rawname, "w")
    CalcRawThetaVals(rawfile, L1, L2, targetxfuncs, targetyfuncs, starts, ends)
    return CleanThetaVals(rawfile.name, file)


def CalcRawThetaVals(rawfile, L1, L2, targetxfuncs, targetyfuncs, starts, ends):
    delim = " "
    L3 = L1 + L2
    th1, th2 = symbols('th1, th2', real=True)
    funcind = 0
    for i in range(len(a)):
        aval = a[i]
        # print(aval)
        if aval >= ends[funcind] and funcind != len(ends) - 1:
            funcind += 1
        eqx = Eq(L2*cos(th2) - L3*sin(th1),
                 targetxfuncs[funcind].subs(t, aval))
        eqy = Eq(L2*sin(th2) + L3*cos(th1),
                 targetyfuncs[funcind].subs(t, aval))
        soln = solve((eqx, eqy), (th1, th2))
        # print(soln, "\n")
        rawfile.write(str(soln[0][0]) + delim + str(soln[0][1]) +
                      delim + str(soln[1][0]) + delim + str(soln[1][1]) + "\n")
    rawfile.close()


def CleanThetaVals(rawfile, file):
    print("Cleaning theta values")
    rS0theta1 = np.loadtxt(rawfile, usecols=0, dtype=float)
    rS0theta2 = np.loadtxt(rawfile, usecols=1, dtype=float)
    rS1theta1 = np.loadtxt(rawfile, usecols=2, dtype=float)
    rS1theta2 = np.loadtxt(rawfile, usecols=3, dtype=float)

    theta1 = np.zeros(len(rS0theta1))
    theta2 = np.zeros(len(rS0theta2))
    for i in range(len(theta1)):
        if (i == 0 or (np.abs(rS1theta1[i] - theta1[i-1]) <= 0.6 and np.abs(rS1theta2[i] - theta2[i-1]) <= 0.6)):
            theta1[i] = rS1theta1[i]
            theta2[i] = rS1theta2[i]
        else:
            theta1[i] = rS0theta1[i]
            theta2[i] = rS0theta2[i]
        file.write(str(theta1[i]) + " " + str(theta2[i]) + "\n")
    print("finished clean")
    file.close()
    return np.array([theta1, theta2])


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


print("do you want to run from existing file?")
if str(input()) == "yes":
    print("input file path: ")
    file = str(input())
    dims = np.loadtxt(file, max_rows=5, dtype=float)
    L1 = dims[0]
    L2 = dims[1]
    L3 = L1 + L2
    Gx = dims[2]
    Gy = dims[3]
    pieces = dims[4]
    datastart = int(len(dims) + pieces)
    theta1 = np.loadtxt(file,
                        usecols=0, skiprows=datastart, dtype=float)
    theta2 = np.loadtxt(file,
                        usecols=1, skiprows=datastart, dtype=float)
    if len(a) != len(theta1):
        a = np.linspace(0, 2*np.pi, len(theta1), dtype=float)
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
    targetxfuncs = targetfuncs[0]
    targetyfuncs = targetfuncs[1]
    starts = targetfuncs[2]
    ends = targetfuncs[3]

    thetavals = CalcThetaVals(
        file, L1, L2, targetxfuncs, targetyfuncs, starts, ends)
    theta1 = thetavals[0]
    theta2 = thetavals[1]
    file.close()

fig, ax = plt.subplots()

# b is the angle relating the center of the rotors to the corresponding linkage point of contact
b_top = np.arctan2(-Gy + L1*np.cos(theta1), -Gx - L1*np.sin(theta1))
c_top = -a + b_top
r_top = ((-Gx - L1*np.sin(theta1))**2 + (-Gy + L1*np.cos(theta1))**2)**0.5

b_bottom = np.arctan2(-Gy + L1*np.cos(theta2), -Gx - L1*np.sin(theta2))
c_bottom = -a + b_bottom
r_bottom = ((-Gx - L1*np.sin(theta2))**2 + (-Gy + L1*np.cos(theta2))**2)**0.5


anim = FuncAnimation(fig, animate,
                     frames=360, interval=100, repeat=true)
plt.show()
