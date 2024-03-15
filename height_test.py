import numpy as np
import matplotlib.pyplot as plt

# Initialize the height map with zeros (default height)
height_map = np.zeros((33, 21))
#height_map.flatten = (693, 1)

def trench(height_map=height_map):
    # Iterate over each point in the grid
    for i in range(height_map.shape[0]):  # For each row
        for j in range(height_map.shape[1]):  # For each column
            # Determine color based on position (using provided logic)
            if (j % 21) < 7:  # Red condition
                height_map[i, j] = 1  # Set height to 1 meter
            elif (j % 21) > 13:  # Green condition
                height_map[i, j] = 1  # Set height to 1 meter
            # Yellow and Blue conditions are ignored for height assignment

#trench()
                
def flatbed(height_map=height_map):
    current_height = 0
    # Iterate over each point in the grid
    for i in range(height_map.shape[0]):  # For each row
        for j in range(height_map.shape[1]):  # For each column
            height_map[i, j] = current_height  # Set height

flatbed()
                
def starting_ascent(height_map):
    max_height = 0.4  # Maximum height at the top edge
    length = 0.8  # Total length in meters
    # Calculate the physical size of each cell in the length direction
    cell_length = length / height_map.shape[0]
    # Calculate midpoint in the length direction
    midpoint_y = length / 2
    # Iterate over each point in the grid
    for i in range(height_map.shape[0]):  # For each row
        # Calculate the physical position of the current cell in the length direction
        y_pos = i * cell_length + cell_length / 2
        # Check if above midpoint (in the top half of the grid)
        if y_pos > midpoint_y:
            # Calculate distance from midpoint in the length direction
            dist_y = abs(midpoint_y - y_pos)
            # Determine the proportional distance from the center to the top edge
            dist_ratio = dist_y / (length / 2)
            # Set height based on distance ratio
            height_map[i, :] = dist_ratio * max_height

#starting_ascent(height_map)

def starting_descent(height_map):
    max_height = 0.4  # Maximum height at the top edge
    length = 0.8  # Total length in meters
    # Calculate the physical size of each cell in the length direction
    cell_length = length / height_map.shape[0]
    # Calculate midpoint in the length direction
    midpoint_y = length / 2
    # Iterate over each point in the grid
    for i in range(height_map.shape[0]):  # For each row
        # Calculate the physical position of the current cell in the length direction
        y_pos = i * cell_length + cell_length / 2
        if y_pos <= midpoint_y:
            height_map[i, :] = max_height
        # Check if above midpoint (in the top half of the grid)
        if y_pos > midpoint_y:
            # Calculate distance from midpoint in the length direction
            dist_y = abs(midpoint_y - y_pos)
            # Determine the proportional distance from the center to the top edge
            dist_ratio = dist_y / (length / 2)
            # Set height based on distance ratio
            height_map[i, :] = max_height - (dist_ratio * max_height)

#starting_descent(height_map)

# Visualize the synthetic height map
plt.imshow(height_map, cmap='viridis', origin='lower', interpolation='none')
plt.colorbar(label='Height (m)')
plt.title('Synthetic Height Map Visualization')
plt.xlabel('X Coordinate')
plt.ylabel('Y Coordinate')
plt.show()

# def starting_ascent(height_map):
#     max_height = 0.4  # Maximum height at the edges
#     width = 0.5  # Total width in meters
#     length = 0.8  # Total length in meters
    
#     # Calculate the physical size of each cell
#     cell_width = width / height_map.shape[1]
#     cell_length = length / height_map.shape[0]

#     # Calculate midpoint coordinates in meters
#     midpoint_x = width / 2
#     midpoint_y = length / 2

#     # Iterate over each point in the grid
#     for i in range(height_map.shape[0]):  # For each row
#         for j in range(height_map.shape[1]):  # For each column
#             # Calculate the physical position of the current cell
#             x_pos = j * cell_width + cell_width / 2
#             y_pos = i * cell_length + cell_length / 2
#             # Calculate distance from midpoint in both directions
#             dist_x = abs(midpoint_x - x_pos)
#             dist_y = abs(midpoint_y - y_pos)
            
#             # Determine the proportional distance from the center to the edge
#             dist_ratio = max(dist_x / (width / 2), dist_y / (length / 2))
            
#             # Set height based on distance ratio
#             height_map[i, j] = dist_ratio * max_height

# starting_ascent(height_map)