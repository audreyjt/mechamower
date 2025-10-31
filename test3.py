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

closest_point = min(hull_points, key=lambda point: np.sqrt(point[0]**2 + point[1]**2))
print("Closest point to (0, 0):", closest_point)



robot_dot, = ax.plot( closest_point[0], closest_point[1], 'ro', markersize=5, label="Robot")




plt.show()

