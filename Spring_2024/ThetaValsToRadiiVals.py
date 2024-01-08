import numpy as np

def CalcRadiiVals(L1, Gx, Gy, theta1, theta2):
    # a is a parametric angle variable in radians
    a = np.linspace(0, 2*np.pi, len(theta1), dtype=float)
    # b is the angle relating the center of the rotors to the corresponding linkage point of contact
    # there is a b for the top and bottom rotors
    b_top = np.arctan2(-Gy + L1*np.cos(theta1), -Gx - L1*np.sin(theta1))
    c_top = -a + b_top
    b_bottom = np.arctan2(-Gy + L1*np.cos(theta2), -Gx - L1*np.sin(theta2))
    c_bottom = -a + b_bottom

    # The radii for the top and bottom rotors is the norm of the vector from the rotor base @ (Gx, Gy)
    # to the top and bottom linkage contact points
    r_top = ((-Gx - L1*np.sin(theta1))**2 + (-Gy + L1*np.cos(theta1))**2)**0.5
    r_bottom = ((-Gx - L1*np.sin(theta2))**2 +
                (-Gy + L1*np.cos(theta2))**2)**0.5
    return a, r_bottom, r_top, c_bottom, c_top