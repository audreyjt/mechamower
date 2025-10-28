import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def rot3_z(theta):
    """3D rotation around z-axis"""
    return np.array([
        [np.cos(theta), -np.sin(theta), 0],
        [np.sin(theta),  np.cos(theta), 0],
        [0, 0, 1]
    ])

def rot3_x(theta):
    """3D rotation around z-axis"""
    return np.array([
        [1, 0, 0],
        [0,  np.cos(theta), -np.sin(theta)],
        [0, np.sin(theta), np.cos(theta)]
    ])

# Define cube vertices (8 corners)
cube = np.array([
    [0, 1, 1, 0, 0, 1, 1, 0],  # x
    [0, 0, 1, 1, 0, 0, 1, 1],  # y
    [0, 0, 0, 0, 1, 1, 1, 1]   # z
])

# Define cube edges (pairs of vertex indices)
edges = [
    (0,1), (1,2), (2,3), (3,0),  # bottom square
    (4,5), (5,6), (6,7), (7,4),  # top square
    (0,4), (1,5), (2,6), (3,7)   # vertical edges
]
# Compute cube center
center = np.mean(cube, axis=1, keepdims=True)

# Translate cube to origin
cube_centered = cube - center

# Rotate cube around Z axis
theta = np.deg2rad(45)
R = rot3_x(theta)
rotated_cube = R @ cube_centered + center

# Plot original and rotated cubes
fig = plt.figure(figsize=(7,7))
ax = fig.add_subplot(111, projection='3d')

# Original cube (blue)
for e in edges:
    ax.plot([cube[0,e[0]], cube[0,e[1]]],
            [cube[1,e[0]], cube[1,e[1]]],
            [cube[2,e[0]], cube[2,e[1]]], 'b-')

# Rotated cube (red)
for e in edges:
    ax.plot([rotated_cube[0,e[0]], rotated_cube[0,e[1]]],
            [rotated_cube[1,e[0]], rotated_cube[1,e[1]]],
            [rotated_cube[2,e[0]], rotated_cube[2,e[1]]], 'r-')

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_box_aspect([1,1,1])  # Equal aspect ratio
plt.show()
