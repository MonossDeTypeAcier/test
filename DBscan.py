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

# Create the plot
plt.figure(figsize=(8, 8))

# Plot the red points
plt.scatter(red_x, red_y, color='red', label='Red Points', s=100)

# Plot the blue points
plt.scatter(blu_x, blu_y, color='blue', label='Blue Points', s=100)

# Set fixed axis limits
plt.xlim(0, 20)
plt.ylim(0, 10)
plt.gca().set_aspect('equal', adjustable='box')

# Add labels and title
plt.title("Red and Blue Points")
plt.xlabel("X-axis")
plt.ylabel("Y-axis")

# Display the grid
plt.grid(True, linestyle='--', alpha=0.6)

# Show the plot
plt.show()


from sklearn.cluster import DBSCAN
debscan = DBSCAN(eps=2, min_samples=2).fit(coordinates)
labels_pred = debscan.labels_
n_clusters_ = len(set(labels_pred)) - (1 if -1 in labels_pred else 0)


unique_labels = set(labels_pred)
core_samples_mask = np.zeros_like(labels_pred, dtype=bool)
core_samples_mask[debscan.core_sample_indices_] = True

colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]
for k, col in zip(unique_labels, colors):
    if k == -1:
        # Black used for noise.
        col = [0, 0, 0, 1]

    class_member_mask = labels_pred == k

    xy = coordinates[class_member_mask & core_samples_mask]
    plt.plot(
        xy[:, 0],
        xy[:, 1],
        "o",
        markerfacecolor=tuple(col),
        markeredgecolor="k",
        markersize=14,
    )

    xy = coordinates[class_member_mask & ~core_samples_mask]
    plt.plot(
        xy[:, 0],
        xy[:, 1],
        "o",
        markerfacecolor=tuple(col),
        markeredgecolor="k",
        markersize=6,
    )

plt.title(f"Estimated number of clusters: {n_clusters_}")
# Set fixed axis limits
plt.xlim(0, 20)
plt.ylim(0, 10)
plt.gca().set_aspect('equal', adjustable='box')

# Add labels
plt.xlabel("X-axis")
plt.ylabel("Y-axis")

# Display the grid
plt.grid(True, linestyle='--', alpha=0.6)

# Show the plot
plt.show()