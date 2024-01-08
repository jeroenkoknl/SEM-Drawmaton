import numpy as np
from sympy import *
import matplotlib.pyplot as plt

# modelled with angular displacement alpha for radii discs, unsigned thetaK

# outputs spiral, in 2nd quadrant


def topFunc(x, r0):
    return sin(8*x) + 0.5*cos(5*x) + r0


def botFunc(x, r0):
    return cos(6*x)*sin(2*x) + r0


def calcThetaKList(r0, rfunc, L1, step):
    alpha1, alpha2, thetaK = symbols('alpha1, alpha2, thetaK', real=True)

    r1 = rfunc(alpha1, r0)
    r2 = rfunc(alpha2, r0)

    alphaList = []
    thetaKList = []
    eq = Eq(r1**2 + r2**2 - 2*r1*r2*cos(alpha2 - alpha1),
            (2*L1**2)*(1 - cos(thetaK)))
    soln = solve(eq, thetaK)[1]
    increments = np.arange(0, 2 * np.pi, step)
    for i in increments:
        a1 = i
        a2 = i + step
        solnCopy = soln
        solnCopy = solnCopy.subs(alpha1, a1)
        dThetaK = solnCopy.subs(alpha2, a2)
        # print(dThetaK)
        alphaList.append(a2)
        absThetaK = 0
        # if (r2.subs(alpha2, a2).evalf() > r1.subs(alpha1, a1).evalf()):
        #     dThetaK = - dThetaK

        if (len(thetaKList) != 0):
            absThetaK = dThetaK + thetaKList[len(thetaKList) - 1]

        thetaKList.append(absThetaK)
    return thetaKList


def calcPenPath(theta1List, theta2List, L1, L2):
    penX = []
    penY = []
    for i in range(len(theta1List)):
        t1 = theta1List[i]
        t2 = theta2List[i]
        penX.append(L2*cos(t2) - (L1 + L2)*sin(t1))
        penY.append(L2*sin(t2) + (L1 + L2)*cos(t1))
    pen = [penX, penY]
    return pen


L1 = 7  # cm
L2 = 15  # cm
r0 = 7.5  # cm
theta1List = calcThetaKList(r0, topFunc, L1, 0.05)
# print("\n\n")
# for i in range(len(theta1List)):
#     print(theta1List[i])
theta2List = calcThetaKList(r0, botFunc, L1, 0.05)
penPath = calcPenPath(theta1List, theta2List, L1, L2)

plot = plt.gca()
plot.set_title("Pen Path")
plot.set_aspect('equal')
xmin = np.double(min(penPath[0]) - 0.25)
xmax = np.double(max(penPath[0]) + 0.25)
ymin = np.double(min(penPath[1]) - 0.25)
ymax = np.double(max(penPath[1]) + 0.25)
plt.xlim(xmin, xmax)
plt.ylim(ymin, ymax)
plt.xlabel('x')
plt.ylabel('y')
plt.plot(penPath[0], penPath[1])
plt.show()
print("finished in original model file")
