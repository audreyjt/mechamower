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

# CLosest and Farthest Points
closest_point = min(hull_points, key=lambda point: np.sqrt(point[0]**2 + point[1]**2))
index_c = np.where((hull_points == closest_point).all(axis=1))[0][0]

farthest_point = max(hull_points, key=lambda point: np.sqrt(point[0]**2 + point[1]**2))
index_f = np.where(hull_points == farthest_point)[0][0]

print("Closest point:", closest_point)
print("Index of closest point:", index_c)
print("Farthest point:", farthest_point)
print("Index of farthest point:", index_f)

#direction
dir_vec = (hull_points[index_c - 1] - closest_point)/2
perp_vec = np.array([-dir_vec[1], dir_vec[0]])
print("Direction:", dir_vec)

minx, miny = np.min(hull_points, axis=0)
maxx, maxy = np.max(hull_points, axis=0)

projections = hull_points @ perp_vec
min_proj, max_proj = projections.min(), projections.max()

spacing = 0.5
lines = []
dir_flag = True

def line_intersections(offset, poly, dir_vec, perp_vec):
    intersections = []
    for i in range(len(poly)):
        p1, p2 = poly[i], poly[(i + 1) % len(poly)]
        d1 = np.dot(p1, perp_vec) - offset
        d2 = np.dot(p2, perp_vec) - offset
        if d1 * d2 <= 0 and d1 != d2:
            s = d1 / (d1 - d2)
            inter = p1 + s * (p2 - p1)
            intersections.append(inter)
    return intersections

offset = min_proj
while offset <= max_proj:
    inters = line_intersections(offset, hull_points, dir_vec, perp_vec)
    inters = sorted(inters, key=lambda p: np.dot(p, dir_vec))
    if len(inters) >= 2:
        for j in range(0, len(inters), 2):
            if j + 1 < len(inters):
                p1, p2 = inters[j], inters[j + 1]
                if dir_flag:
                    line = np.array([p1, p2])
                else:
                    line = np.array([p2, p1])
                lines.append(line)
                dir_flag = not dir_flag
    offset += spacing

path_points = np.concatenate(lines)
ax.plot(path_points[:, 0], path_points[:, 1], 'b--', lw=1, alpha=0.6, label="Path")


#time
t = 0
dt = 0.1
speed = 0.1

#robot
pos = np.array(path_points[0])
(robot_dot,) = ax.plot( pos[0], pos[1], 'ro', markersize=5, label="Robot")

#loop
for target in path_points[1:]:
    dir_vec_move = target - pos
    distance = np.linalg.norm(dir_vec_move)
    if distance == 0:
        continue
    dir_vec_move /= distance
    while np.linalg.norm(target - pos) > speed:
        pos += dir_vec_move * speed
        robot_dot.set_data([pos[0]], [pos[1]])
        plt.pause(dt)
    pos = target

plt.show()

