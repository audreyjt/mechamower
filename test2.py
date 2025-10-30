import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon
from scipy.spatial import ConvexHull

size_min = 1
size_limit= 9
np.random.seed(12)
poly_points = np.clip((np.random.rand(8, 2) * 11), 0, size_limit)

fig, ax = plt.subplots()
hull = ConvexHull(poly_points)
hull_points=poly_points[hull.vertices]
poly = Polygon(hull_points, closed =True, facecolor='green')
ax.set_xlim(0, 10) # Set x-axis limits
ax.set_ylim(0, 10) # Set y-axis limits
ax.add_patch(poly)
plt.show()
