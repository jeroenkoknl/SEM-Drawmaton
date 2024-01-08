import numpy as np
import sympy as sp
from sympy import *
from sympy.plotting import plot_parametric
from sympy.parsing.sympy_parser import parse_expr
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import pandas as pd

a = np.linspace(0, 2*np.pi, 100, dtype=float)

print(np.linspace(0, 2*np.pi*(3/4), 4, dtype=float))
print(np.linspace(2*np.pi*(1/4), 2*np.pi, 4, dtype=float))
print(np.linspace(0,0,1,dtype=float))

print("do you want to run from existing file?")
if str(input()) == "yes":
    print("input file path: ")
    file = str(input())
    dims = np.loadtxt(file, max_rows=4, dtype=float)
    # print(dims)
    L1 = dims[0]
    L2 = dims[1]
    L3 = L1 + L2
    Gx = dims[2]
    Gy = dims[3]
    t = symbols('t')
    xfuncs = np.loadtxt(file, skiprows=4, delimiter=",", usecols=0, dtype=str)
    yfuncs = np.loadtxt(file, skiprows=4, delimiter=",", usecols=1, dtype=str)
    starts = np.loadtxt(file, skiprows=4, delimiter=",",
                        usecols=2, dtype=float)
    ends = np.loadtxt(file, skiprows=4, delimiter=",", usecols=3, dtype=float)

    targetxfuncs = []
    targetyfuncs = []
    for i in range(len(xfuncs)):
        # print(parse_expr(xfuncs[i]), parse_expr(yfuncs[i]), parse_expr(starts[i]), parse_expr(ends[i]))
        targetxfuncs.append(parse_expr(xfuncs[i]))
        targetyfuncs.append(parse_expr(yfuncs[i]))

# plot((xfuncs[0], (t, starts[0], ends[0])), (xfuncs[1], (t, starts[1], ends[1])),
#      (xfuncs[2], (t, starts[2], ends[2])), (xfuncs[3], (t, starts[3], ends[3])))
print(targetxfuncs[1].subs(t, 0))
th1, th2 = symbols('th1, th2', real=True)
funcind = 0
for i in range(len(a)):
    aval = a[i]
    print(aval)
    if aval >= ends[funcind] and funcind != len(ends) - 1:
        print("incrementing funct")
        funcind += 1
    eqx = Eq(L2*cos(th2) - L3*sin(th1), targetxfuncs[funcind].subs(t, aval))
    eqy = Eq(L2*sin(th2) + L3*cos(th1), targetyfuncs[funcind].subs(t, aval))
    soln = solve((eqx, eqy), (th1, th2))
    print(soln, "\n")
print(funcind)
