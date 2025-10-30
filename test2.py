import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon
from scipy.spatial import ConvexHull

poly_points = np.random.rand(6, 2) * 10

fig, ax = plt.subplots()
hull = ConvexHull(poly_points)
hull_points=poly_points[hull.vertices]
poly = Polygon(hull_points, closed =True, facecolor='green')
ax.set_xlim(0, 10) # Set x-axis limits
ax.set_ylim(0, 10) # Set y-axis limits
ax.add_patch(poly)
plt.show()
