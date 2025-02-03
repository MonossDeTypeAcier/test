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


from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=2)
kmeans.fit(coordinates)
y_kmeans = kmeans.predict(coordinates)

plt.scatter(coordinates[:, 0], coordinates[:, 1], c=y_kmeans, s=50, cmap='viridis')

centers = kmeans.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200);

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


plt.show()