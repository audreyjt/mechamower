import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon
from scipy.spatial import ConvexHull

def plot_spokes(spokes_left_obj=None, spokes_right_obj=None, wheel_rad=3, x_0=0, y_0=0, n_spokes=7, offset_theta_l=0, offset_theta_r=0):

    spokes_x =lambda offset: x_0 + wheel_rad * np.cos(np.linspace(0 + offset, 2 * np.pi + offset, n_spokes))
    spokes_y =lambda offset: y_0 + wheel_rad * np.sin(np.linspace(0 + offset, 2 * np.pi + offset, n_spokes))
    if spokes_left_obj:
        cur_spokes_x = spokes_x(offset_theta_l)
        cur_spokes_y = spokes_y(offset_theta_l)
        for i in range(n_spokes):
            spokes_left_obj[i].set_data([cur_spokes_x[i] * .2, cur_spokes_x[i]], [cur_spokes_y[i] * .2, cur_spokes_y[i]])

    if spokes_right_obj:
        cur_spokes_x = spokes_x(offset_theta_r)
        cur_spokes_y = spokes_y(offset_theta_r)
        for i in range(n_spokes):
            spokes_right_obj[i].set_data([cur_spokes_x[i]*.2,cur_spokes_x[i]] , [cur_spokes_y[i]*.2, cur_spokes_y[i]])

    if spokes_left_obj is None and spokes_right_obj is None:
        spokes_x = x_0 + wheel_rad * np.cos(np.linspace(0, 2 * np.pi, n_spokes))
        spokes_y = y_0 + wheel_rad * np.sin(np.linspace(0, 2 * np.pi, n_spokes))
        left_spokes = []
        right_spokes = []
        for i in range(n_spokes):
            (left_spoke,) = ax[1].plot([spokes_x[i]*.2, spokes_x[i]], [spokes_y[i]*.2,spokes_y[i]] ,color="k", lw=4.0)
            (right_spoke,) = ax[2].plot([spokes_x[i] * .2, spokes_x[i]], [spokes_y[i] * .2, spokes_y[i]], color="k", lw=4.0)
            left_spokes.append(left_spoke)
            right_spokes.append(right_spoke)
        return left_spokes, right_spokes


def plot_wheels(wheel_radius):
    circle_x = 0
    circle_y = 0
    path_theta = np.linspace(0, 2 * np.pi, 600)
    path_x = circle_x + wheel_radius * np.cos(path_theta)
    path_y = circle_y + wheel_radius * np.sin(path_theta)
    ax[1].plot(path_x, path_y, lw=6.0, c= "k"),ax[1].plot(path_x*.2, path_y*.2, lw=6.0, c="k"), ax[1].set_title("Left Wheel")
    ax[2].plot(path_x, path_y, lw=6.0, c = "k"), ax[2].plot(path_x*.2, path_y*.2, lw=6.0, c="k"), ax[2].set_title("Right Wheel")
    return plot_spokes(wheel_rad=wheel_radius, x_0=circle_x, y_0=circle_y, n_spokes=7)




poly_points = np.random.rand(6, 2) * 10

fig, ax = plt.subplots(1,3, figsize = (15,5))
hull = ConvexHull(poly_points)
hull_points=poly_points[hull.vertices]
poly = Polygon(hull_points, closed =True, facecolor='green')
ax[0].set_xlim(0, 10) # Set x-axis limits
ax[0].set_ylim(0, 10) # Set y-axis limits
ax[0].add_patch(poly)

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
ax[0].plot(path_points[:, 0], path_points[:, 1], 'b--', lw=1, alpha=0.6, label="Path")
left_spks, right_spks = plot_wheels(wheel_radius=3)

#time
t = 0
dt = 0.1
speed = 0.1

#robot
pos = np.array(path_points[0])
(robot_dot,) = ax[0].plot( pos[0], pos[1], 'ro', markersize=5, label="Robot")

#loop
for target in path_points[1:]:

    dir_vec_move = target - pos
    distance = np.linalg.norm(dir_vec_move)
    if distance == 0:
        continue
    dir_vec_move /= distance
    while np.linalg.norm(target - pos) > speed:
        #if it's far away go this speed, else go small_speed
        t += dt
        pos += dir_vec_move * speed
        robot_dot.set_data([pos[0]], [pos[1]])
        omega_l = 2
        omega_r = 5
        plot_spokes(spokes_left_obj=left_spks, spokes_right_obj=right_spks, offset_theta_l=t*omega_l, offset_theta_r=t*omega_r)



        plt.pause(dt)

    # pause to play the turning animation,
    # we need to find out what angle the new vector makes with the old vector, so we can calculate the angle we need to
    # turn the wheels
    pos = target

plt.show()


