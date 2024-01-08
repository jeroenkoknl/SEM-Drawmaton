import matplotlib.pyplot as plt
import numpy as np
import matplotlib
matplotlib.use("svg")

t = np.linspace(0, 2*np.pi, 150)
r = 5
print(r*np.cos(t), np.sin(t))
fig, ax = plt.subplots()

ax.plot(r*np.cos(t), r*np.sin(t))
ax.set_aspect('equal')
ax.axis(False)
plt.savefig("testme.svg", format='svg')
