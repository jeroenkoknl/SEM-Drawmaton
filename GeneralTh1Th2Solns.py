import numpy as np
from sympy import *
import matplotlib.pyplot as plt
from sympy.plotting import plot_parametric

# solves for cos(theta1), sin(theta1), cos(theta2), sin(theta2) as general functions of t
# target hypotrochoid function = <1.4*cos(t) - 2.4*cos(7*t) + 18, 1.4*sin(t) - 2.4*sin(7*t) + 18>
# configured for hypotrochoid currently
# circle \left(3\sin\left(t\right)\ +\ 18,\ 3\cos\left(t\right)\ +\ 18\right)

L1 = 7.6
L2 = 15
L3 = L1 + L2

t, cos_t1, sin_t1, cos_t2, sin_t2, Rx, Ry = symbols(
    't, cos_t1, sin_t1, cos_t2, sin_t2, Rx, Ry', real=True)
eq1 = Eq(L2*cos_t2 - L3*sin_t1, Rx)
eq2 = Eq(L2*sin_t2 + L3*cos_t1, Ry)
eq3 = Eq(cos_t1**2 + sin_t1**2, 1)
eq4 = Eq(cos_t2**2 + sin_t2**2, 1)
soln = solve((eq1, eq2, eq3, eq4), (cos_t1, sin_t1, cos_t2, sin_t2))
# print(len(soln), "\n\n\n", soln[0], "\n\n\n", soln[1])

soln0 = soln[0]
soln1 = soln[1]

targetx = 1.4*cos(t) - 2.4*cos(7*t) + 18
targety = 1.4*sin(t) - 2.4*sin(7*t) + 18

# soln00 = soln0[0].subs(Rx, targetx)
# soln00 = soln00.subs(Ry, targety)
# print(soln00, "\n\n\n")

# plot(soln00, (t, 0, 2*pi))  # soln00 has no match?


soln10 = soln1[0].subs(Rx, targetx)
soln10 = soln10.subs(Ry, targety)
# print(soln10, "\n\n\n")

plot(soln10, (t, 0, 2*pi))  # soln10 is correct cos(theta1) for hypotrochoid

soln11 = soln1[1].subs(Rx, targetx)
soln11 = soln11.subs(Ry, targety)
# print(soln11, "\n\n\n")

# plot(soln11, (t, 0, 2*pi))  # soln11 is correct sin(theta1) for hypotrochoid

soln12 = soln1[2].subs(Rx, targetx)
soln12 = soln12.subs(Ry, targety)
# print(soln12, "\n\n\n")

# plot(soln12, (t, 0, 2*pi))  # soln12 is correct cos(theta2) for hypotrochoid

soln13 = soln1[3].subs(Rx, targetx)
soln13 = soln13.subs(Ry, targety)
# print(soln13, "\n\n\n")

# plot(soln13, (t, 0, 2*pi))  # soln13 is correct sin(theta2) for hypotrochoid

cos_theta1 = soln10
sin_theta1 = soln11
cos_theta2 = soln12
sin_theta2 = soln13
