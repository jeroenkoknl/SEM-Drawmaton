import numpy as np
from sympy import *

def ParametricToXY(targetxfuncs, targetyfuncs, starts, ends, datacount):
    t = symbols('t')
    tvals = np.linspace(0, 2*np.pi, datacount)
    xcoords = np.zeros(len(tvals))
    ycoords = np.zeros(len(tvals))
    # Keep track of which x,y functions to evaluate depending on which domain the current tval lies in
    funcind = 0
    for i in range(len(tvals)):
        tval = tvals[i]
        # Increment the current function if we have entered the succeeding domain
        if tval >= ends[funcind] and funcind != len(ends) - 1:
            funcind += 1
        xcoords[i] = targetxfuncs[funcind].subs(t, tval)
        ycoords[i] = targetyfuncs[funcind].subs(t, tval)
    return [xcoords, ycoords]