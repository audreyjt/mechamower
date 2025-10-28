import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Simulation parameters
lawn_width = 10       # meters
lawn_height = 5       # meters
sweep_width = 1.0     # distance between passes
overlap = 0.1         # overlap between passes
speed = 0.5         # movement per frame

# Derived parameters
effective_sweep = sweep_width - overlap
num_passes = int(np.ceil(lawn_height / effective_sweep))

# Robot state (vector-based)
pos = np.array([0.0, 0.0])
dir_vec = np.array([1.0, 0.0])
direction = 1  # 1 = right, -1 = left
current_pass = 0

# Path storage
path_x, path_y = [pos[0]], [pos[1]]

# Setup plot
fig, ax = plt.subplots()
ax.set_xlim(0, lawn_width)
ax.set_ylim(0, lawn_height)
ax.set_aspect('equal')
ax.set_title("Lawnmower Robot Simulation")

robot_dot, = ax.plot([], [], 'ro', label="Robot")
path_line, = ax.plot([], [], 'b-', linewidth=1, alpha=0.6, label="Path")
ax.legend(loc='upper right')

# Animation update function
def update(frame):
    global pos, dir_vec, direction, current_pass

    # Stop condition: finished all passes
    if current_pass >= num_passes:
        return robot_dot, path_line

    # Move forward
    pos += dir_vec * speed

    # Check for boundary crossing
    if (direction == 1 and pos[0] >= lawn_width) or (direction == -1 and pos[0] <= 0):
        current_pass += 1
        if current_pass < num_passes:
            pos[1] = min(lawn_height, current_pass * effective_sweep)
            direction *= -1
            dir_vec = np.array([direction, 0.0])
        else:
            # Finished mowing
            pos[0] = np.clip(pos[0], 0, lawn_width)
            dir_vec[:] = 0.0

    # Record path
    path_x.append(pos[0])
    path_y.append(pos[1])

    # Update visuals
    robot_dot.set_data([pos[0]], [pos[1]])
    path_line.set_data(path_x, path_y)
    return robot_dot, path_line

# Create animation
ani = FuncAnimation(fig, update, frames=2000, interval=30, blit=True, repeat=False)

plt.show()
