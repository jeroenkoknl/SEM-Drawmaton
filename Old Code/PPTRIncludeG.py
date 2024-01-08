import numpy as np
from sympy import *
import matplotlib.pyplot as plt


# modelled with angular displacement alpha for radii discs, unsigned thetaK
# # target hypotrochoid function = <1.4*cos(t) - 2.4*cos(7*t) + 18, 1.4*sin(t) - 2.4*sin(7*t) + 18>
# circle \left(3\sin\left(t\right)\ +\ 18,\ 3\cos\left(t\right)\ +\ 18\right)

def calcTheta12Lists(targetXfunc, targetYfunc, L1, L2, step):
    t, theta1, theta2 = symbols('t, theta1, theta2', real=True)

    PenX = L2*cos(theta2) - (L1 + L2)*sin(theta1)
    PenY = L2*sin(theta2) + (L1 + L2)*cos(theta1)

    tvals = np.arange(0, 2*np.pi, step)
    theta1vals = np.zeros(len(tvals))
    theta2vals = np.zeros(len(tvals))

    print(tvals.shape)
    for i in range(len(tvals)):
        tval = tvals[i]
        eqx = Eq(PenX, targetXfunc.subs(t, tval))
        eqy = Eq(PenY, targetYfunc.subs(t, tval))
        soln = solve((eqx, eqy), (theta1, theta2))[1]
        # print(soln[0])
        theta1vals[i] = (soln[0].evalf())
        theta2vals[i] = (soln[1].evalf())
    return np.array([theta1vals, theta2vals])


def calcDThetaKList(thetaKList):
    dThetaKList = np.zeros(len(thetaKList))
    for i in range(len(thetaKList)):
        if (i == 0):
            dThetaKList[i] = (thetaKList[0])
        else:
            dThetaKList[i] = (thetaKList[i] - thetaKList[i-1])
        # print(dThetaKList[i])

    dThetaKList[1] = dThetaKList[2] - (dThetaKList[5] - dThetaKList[2]) / 3
    dThetaKList[0] = dThetaKList[1] - (dThetaKList[4] - dThetaKList[1]) / 3

    return dThetaKList


def calcRvalList(thetaKList, r0, L1, L2, step):
    # # piston approximation
    # dr, dThetaK = symbols('dr, dThetaK', real=True)
    # eq = Eq(dr**2, (2*L1**2)*(1 - cos(dThetaK)))
    # #eqCopy = eqCopy.subs(dThetaK, dThetaKList[0])
    # soln = solve(eq, dr)[1]
    # # soln0 = soln[0].subs(dThetaK, dThetaKList[0])
    # # soln1 = soln[1].subs(dThetaK, dThetaKList[0])
    # rvalList = []
    # for dThetaval in dThetaKList:
    #     rval = 0
    #     drval = soln.subs(dThetaK, dThetaval)
    #     if (dThetaval < 0):
    #         drval *= -1
    #     if len(rvalList) == 0:
    #         rvalList.append(r0 + drval)
    #     else:
    #         rvalList.append(rvalList[len(rvalList) - 1] + drval)
    #     # print(drval, rvalList[len(rvalList) - 1])
    # return rvalList
    Ck = np.array([-L1*sin(thetaKList), L1*cos(thetaKList)])
    G = np.array([-2, 5])
    r = Ck - G
    R = np.abs(r)
    return R


def calcXYList(rvalList, step):
    thetaList = np.arange(0, 2*np.pi, step)
    xList = np.zeros(len(thetaList))
    yList = np.zeros(len(thetaList))
    for i in range(len(thetaList)):
        xList[i] = (rvalList[i] * cos(thetaList[i]))
        yList[i] = (rvalList[i] * sin(thetaList[i]))
    return [xList, yList]


t = symbols('t', real=true)
L1 = 7.6  # cm
L2 = 15  # cm
r0 = 8  # cm
step = 0.05

targetXfunc = 1.4*cos(t) - 2.4*cos(7*t) + 18
targetYfunc = 1.4*sin(t) - 2.4*sin(7*t) + 18
# targetXfunc = 4*cos(t) + L1
# targetYfunc = 4*sin(t) + L1
thetavals = calcTheta12Lists(targetXfunc, targetYfunc, L1, L2, step)
theta1vals = thetavals[0]
# theta2vals = thetavals[1]
print("\n\n")
# dTheta1vals = calcDThetaKList(theta1vals)
# dTheta2vals = calcDThetaKList(theta2vals)

topDiskRvals = calcRvalList(theta1vals, r0, L1, L2, step)
# bottomDiskRvals = calcRvalList(dTheta2vals, r0, L1, L2, step)
topDiskXY = calcXYList(topDiskRvals, step)
# botDiskXY = calcXYList(bottomDiskRvals, step)


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

# plt.plot(botDiskXY[0], botDiskXY[1])
plt.plot(topDiskXY[0], topDiskXY[1])
refcirc = plt.Circle((0, 0), radius=1, color='blue', fill=False)
# bttmouter = plt.Circle((0, 0), radius=2.45, color='green', fill=False)
# bttminner = plt.Circle((0, 0), radius=2, color='green', fill=False)
# topouter = plt.Circle((0, 0), radius=2.325, color='red', fill=False)
# topinner = plt.Circle((0, 0), radius=2.125, color='red', fill=False)
plt.gca().add_artist(refcirc)
# plt.gca().add_artist(bttmouter)
# plt.gca().add_artist(bttminner)
# plt.gca().add_artist(topouter)
# plt.gca().add_artist(topinner)

plt.legend()
plt.show()
plt.savefig("myimg.svg")
