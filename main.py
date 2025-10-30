import matplotlib.pyplot as plt
import numpy as np


rl = 5
rw = 10
rect = np.array([[-rl/2, -rl/2, rl/2, rl/2], [-rw/2, rw/2, rw/2, rw/2]])


fig, ax = plt.subplots()
ax.plot(rect[0], rect[1])
plt.show()