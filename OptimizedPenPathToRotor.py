import numpy as np
from sympy import *
import matplotlib.pyplot as plt

# modelled with angular displacement alpha for radii discs, unsigned thetaK
# target hypotrochoid function = <1.4*cos(t) - 2.4*cos(7*t) + 18, 1.4*sin(t) - 2.4*sin(7*t) + 18>


def calcTheta12Lists(targetXfunc, targetYfunc, L1, L2, step):
    t, theta1, theta2 = symbols('t, theta1, theta2', real=True)

    PenX = L2*cos(theta2) - (L1 + L2)*sin(theta1)
    PenY = L2*sin(theta2) + (L1 + L2)*cos(theta1)

    tvals = np.arange(0, 2*np.pi, step)
    theta1vals = [solve((Eq(PenX, targetXfunc.subs(t, tval)), Eq(
        PenY, targetYfunc.subs(t, tval))), (theta1, theta2))[0][0].evalf() for tval in tvals]
    theta2vals = [solve((Eq(PenX, targetXfunc.subs(t, tval)), Eq(
        PenY, targetYfunc.subs(t, tval))), (theta1, theta2))[0][1].evalf() for tval in tvals]

    return theta1vals, theta2vals


def calcDThetaKList(thetaKList):
    return [thetaKList[0]] + [thetaKList[i] - thetaKList[i-1] for i in range(1, len(thetaKList))] \
        + [thetaKList[1] - (thetaKList[5] - thetaKList[2]) / 3] \
        + [thetaKList[0] - (thetaKList[4] - thetaKList[1]) / 3]


def calcRvalList(dThetaKList, r0, L1, L2, step):
    # piston approximation
    dr, dThetaK = symbols('dr, dThetaK', real=True)
    eq = Eq(dr**2, (2*L1**2)*(1 - cos(dThetaK)))
    soln = solve(eq, dr)[1]
    rvalList = [r0 + (soln.subs(dThetaK, dTheta).subs(dTheta, dThetaval) if dThetaval >= 0 else -soln.subs(
        dThetaK, -dTheta).subs(dTheta, -dThetaval)) for dThetaval, dTheta in zip(dThetaKList, [0] + dThetaKList[:-1])]
    return rvalList


def calcXYList(rvalList, step):
    thetaList = np.arange(0, 2*np.pi, step)
    xList = [rval * cos(theta) for rval, theta in zip(rvalList, thetaList)]
    yList = [rval * sin(theta) for rval, theta in zip(rvalList, thetaList)]
    return [xList, yList]


t = symbols('t', real=true)
L1 = 7  # cm
L2 = 15  # cm
r0 = 7.5  # cm
step = 0.02
targetXfunc = 4*sin(t) + 18
targetYfunc = 4*cos(t) + 18
thetavals = calcTheta12Lists(targetXfunc, targetYfunc, L1, L2, step)
#theta1vals = thetavals[0]
theta2vals = thetavals[1]
print("\n\n")
#dTheta1vals = calcDThetaKList(theta1vals)
dTheta2vals = calcDThetaKList(theta2vals)

#topDiskRvals = calcRvalList(dTheta1vals, r0, L1, L2, step)
bottomDiskRvals = calcRvalList(dTheta2vals, r0, L1, L2, step)
#topDiskXY = calcXYList(topDiskRvals, step)
botDiskXY = calcXYList(bottomDiskRvals, step)


# tvals = np.arange(0, 2*np.pi, step)
plot = plt.gca()
# plot.set_title(
#    "Bottom disk profile for hypotrochoid output <1.4*cos(t) - 2.4*cos(7*t) + 18, 1.4*sin(t) - 2.4*sin(7*t) + 18>")
plot.set_aspect('equal')
# xmin = 0
# xmax = np.double(2*np.pi)
# ymin = -0.05
# ymax = 0.05
# plt.xlim(xmin, xmax)
# #plt.ylim(ymin, ymax)
plt.xlabel('x')
plt.ylabel('y')
plt.plot(botDiskXY[0], botDiskXY[1])

# plt.plot(topDiskXY[0], topDiskXY[1], label="Top Rotor profile")
circle = plt.Circle((0, 0), radius=1)
plt.gca().add_artist(circle)
plt.legend()
plt.show()
plt.savefig("myimg.svg")
