import numpy as np
from sympy import *
from sympy.plotting import plot_parametric

import matplotlib.pyplot as plt

t = symbols('t', real=true)

L1 = 2  # cm
L2 = 6  # cm
G = np.array([-15, 15])
# step = 0.05
# tvals = np.arange(0, 2*pi, step)
# r0 = 7.87

# theta1 = 3*t
theta2 = 4*t
# dtheta1 = 3*step
# dtheta2 = 4*step


# r1, r2, dtheta = symbols('r1, r2, dtheta', real=True)
# eq = Eq((r2-r1)**2, (2*L1**2)*(1-cos(dtheta2)))
# soln = solve(eq, r2)
# print(soln)
# for tval in tvals:
#     r0 = soln[1].subs(r1, r0)
#     print(r0)

# Ghat = G/np.linalg.norm(G)
# C2oft = np.array([-L1*sin(theta2), L1*cos(theta2)])
# GC2oft = C2oft - G
# R2oft = np.dot(GC2oft, -Ghat)*Ghat
# print(R2oft[0], "\n", R2oft[1])

# Ghat = G/np.linalg.norm(G)
C2oft = np.array([-L1*sin(theta2), L1*cos(theta2)])
GC2oft = C2oft - G
GC2hatoft = GC2oft / (GC2oft[0]**2 + GC2oft[1]**2)**0.5
R2oft = -np.dot(GC2hatoft, G)*GC2hatoft
print(R2oft[0])
plot(R2oft[0], (t, 0, 2*pi))
