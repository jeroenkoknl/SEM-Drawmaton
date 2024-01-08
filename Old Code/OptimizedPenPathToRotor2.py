import numpy as np
from sympy import symbols, cos, sin, Eq, solve, lambdify
import matplotlib.pyplot as plt


def calcRotors(L1, L2, r0, targetXfunc, targetYfunc, step):
    t = symbols('t', real=True)
    theta1 = 2 * np.arcsin((L1 + L2 - r0) / (2 * L1))
    theta2 = 2 * np.arcsin((L1 - L2 + r0) / (2 * L2))

    # Convert target functions to callable functions using lambdify
    x_func = lambdify(t, targetXfunc, "numpy")
    y_func = lambdify(t, targetYfunc, "numpy")

    # Calculate theta values for each angle
    t_vals = np.arange(0, 2 * np.pi, step)
    theta1_vals = np.full_like(t_vals, theta1)
    theta2_vals = np.full_like(t_vals, theta2)
    for i in range(len(t_vals)):
        eqx = Eq(L2 * cos(theta2) - (L1 + L2) * sin(theta1), x_func(t_vals[i]))
        eqy = Eq(L2 * sin(theta2) + (L1 + L2) * cos(theta1), y_func(t_vals[i]))
        soln = solve((eqx, eqy), (theta1, theta2))
        print(soln)
        # theta1_vals[i], theta2_vals[i] = soln[0][0], soln[0][1]

    # Calculate piston displacement
    d_theta_2 = np.diff(theta2_vals)
    d_theta_2 = np.concatenate(([theta2_vals[0]], d_theta_2))
    dr_vals_bot = np.sqrt(2 * L1 ** 2 * (1 - np.cos(d_theta_2)))

    d_theta_1 = np.diff(theta1_vals)
    d_theta_1 = np.concatenate(([theta1_vals[0]], d_theta_1))
    dr_vals_top = np.sqrt(2 * L1 ** 2 * (1 - np.cos(d_theta_1)))

    # Calculate radii for each angle
    r_vals_bot = np.zeros_like(t_vals)
    r_vals_bot[0] = r0
    r_vals_top = np.zeros_like(t_vals)
    r_vals_top[0] = r0
    for i in range(len(dr_vals_bot)):
        r_vals_bot[i+1] = r_vals_bot[i] + \
            dr_vals_bot[i] * np.sign(d_theta_2[i])
        r_vals_top[i+1] = r_vals_top[i] + \
            dr_vals_top[i] * np.sign(d_theta_1[i])

    # Calculate x and y coordinates
    x_vals_bot = r_vals_bot * np.cos(t_vals)
    y_vals_bot = r_vals_bot * np.sin(t_vals)
    x_vals_top = r_vals_top * np.cos(t_vals)
    y_vals_top = r_vals_top * np.sin(t_vals)

    return x_vals_bot, y_vals_bot, x_vals_top, y_vals_top


t = symbols('t', real=True)
L1 = 7  # cm
L2 = 15  # cm
r0 = 7.5  # cm
step = 0.025
targetXfunc = 1.4*cos(t) - 2.4*cos(7*t) + 18
targetYfunc = 1.4*sin(t) - 2.4*sin(7*t) + 18

rotorXYs = calcRotors(L1, L2, r0, targetXfunc, targetYfunc, step)

plot = plt.gca()
plot.set_title(
    "Optimized Rotor profiles for circle output <3sin(t) + 18, 3cos(t) + 18>")
plt.xlabel('x')
plt.ylabel('y')
plt.xlim(-10, 10)
plt.ylim(-10, 10)
plt.plot(rotorXYs[0], rotorXYs[1], label="Bottom Rotor profile")
plt.plot(rotorXYs[2], rotorXYs[3], label="Top Rotor profile")
plt.legend()
plt.show()
