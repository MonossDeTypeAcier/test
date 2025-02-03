import matplotlib.pyplot as plt
import numpy as np
import random

nb_point = 30

tab_points = []

for i in range(nb_point):
    x_red = np.random.uniform(3.0, 7.0)
    y_red = np.random.uniform(3.0, 7.0)
    tab_points.append((x_red, y_red, 1))  # 1 for red

    x_blu = np.random.uniform(13.0, 17.0)
    y_blu = np.random.uniform(3.0, 7.0)
    tab_points.append((x_blu, y_blu, 0))  # 0 for blue

# Shuffle the points to randomize the order
random.shuffle(tab_points)

# Separate the coordinates and labels
coordinates = np.array([(x, y) for x, y, label in tab_points])
labels = np.array([label for x, y, label in tab_points])

print("Coordinates array:")
print(coordinates)
print("Labels array:")
print(labels)

# Extract x and y coordinates for plotting
red_x, red_y = coordinates[labels == 1].T
blu_x, blu_y = coordinates[labels == 0].T

# Create the figure and a set of subplots
fig, ax = plt.subplots(1, 2, figsize=(16, 8))

# Plot the red points on the first subplot
ax[0].scatter(red_x, red_y, color='red', label='Red Points', s=100)

# Plot the blue points on the first subplot
ax[0].scatter(blu_x, blu_y, color='blue', label='Blue Points', s=100)

# Set fixed axis limits for the first subplot
ax[0].set_xlim(0, 20)
ax[0].set_ylim(0, 10)
ax[0].set_aspect('equal', adjustable='box')

# Add labels and title to the first subplot
ax[0].set_title("Red and Blue Points")
ax[0].set_xlabel("X-axis")
ax[0].set_ylabel("Y-axis")

# Display the grid on the first subplot
ax[0].grid(True, linestyle='--', alpha=0.6)

from sklearn.neighbors import NearestNeighbors
nbrs = NearestNeighbors(n_neighbors=2).fit(coordinates)
distances, indices = nbrs.kneighbors(coordinates)

print("Distances:")
print(distances)
print("Indices:")
print(indices)

# Plot the points on the second subplot
ax[1].scatter([x[0] for x in coordinates], [x[1] for x in coordinates], color='green', s=100)

# Draw lines between each point and its nearest neighbors on the second subplot
for i, (point, neighbors) in enumerate(zip(coordinates, indices)):
    for neighbor in neighbors:
        ax[1].plot([point[0], coordinates[neighbor][0]], [point[1], coordinates[neighbor][1]], 'k--')

# Add labels and title to the second subplot
ax[1].set_title("Nearest Neighbors")
ax[1].set_xlabel("X-axis")
ax[1].set_ylabel("Y-axis")

# Display the grid on the second subplot
ax[1].grid(True, linestyle='--', alpha=0.6)

# Set fixed axis limits for the second subplot
ax[1].set_xlim(0, 20)
ax[1].set_ylim(0, 10)
ax[1].set_aspect('equal', adjustable='box')

# Show the plot
plt.show()