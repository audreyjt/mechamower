import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


lawn_width = 10
lawn_height = 5
robot_width = 1.2
overlap = 0.2
speed = 0.5
blade_diam = 1.0

effective_sweep = blade_diam - overlap
num_passes = int(np.ceil(lawn_height / effective_sweep))

pos = np.array([0.0, 0.0])
dir_vec = np.array([1.0, 0.0])
direction = 1  # 1 = right, -1 = left
current_pass = 0

path_x, path_y = [pos[0]], [pos[1]]

fig, ax = plt.subplots()
ax.set_xlim(0, lawn_width)
ax.set_ylim(0, lawn_height)
ax.set_aspect('equal')
ax.set_title("Lawnmower Robot Simulation")

robot_dot, = ax.plot([], [], 'ro', markersize=5, label="Robot", zorder=3)
robot_blade, = ax.plot([], [], 'ro', markersize=blade_diam * 36, alpha=0.3, label="Robot", zorder=2)
path_line, = ax.plot([], [], 'b-', linewidth= robot_width * 36 , alpha=0.3, label="Path", zorder = 1)



def update(_):
    global pos, dir_vec, direction, current_pass

    if current_pass >= num_passes:
        return robot_dot, path_line, robot_blade

    pos += dir_vec * speed

    if (direction == 1 and pos[0] >= lawn_width) or (direction == -1 and pos[0] <= 0):
        current_pass += 1
        if current_pass < num_passes:
            pos[1] = min(lawn_height, current_pass * effective_sweep)
            direction *= -1
            dir_vec = np.array([direction, 0.0])
        else:
            pos[0] = np.clip(pos[0], 0, lawn_width)
            dir_vec[:] = 0.0

    path_x.append(pos[0])
    path_y.append(pos[1])

    robot_dot.set_data([pos[0]], [pos[1]])
    robot_blade.set_data([pos[0]], [pos[1]])
    path_line.set_data(path_x, path_y)
    return robot_dot, path_line, robot_blade

ani = FuncAnimation(fig, update, frames=2000, interval=30, blit=True, repeat=False)

plt.show()
