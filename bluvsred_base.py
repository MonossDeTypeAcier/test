import matplotlib.pyplot as plt
import numpy as np

nb_point = 30

tab_red = []
tab_blu = []

for i in range(nb_point) :
  x_red = np.random.uniform(3.0,7.0)
  y_red = np.random.uniform(3.0,7.0)
  tab_red.append((x_red, y_red, ))


  x_blu = np.random.uniform(13.0,17.0)
  y_blu = np.random.uniform(3.0,7.0)
  tab_blu.append((x_blu, y_blu))

print("Tableau de DATA de point rouge : ")
print(tab_red)
print("------------------------------------------")
print("Tableau de DATA de point bleu : ")
print(tab_blu)
print("------------------------------------------")


# Extract x and y coordinates
red_x, red_y = zip(*tab_red)
blu_x, blu_y = zip(*tab_blu)

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