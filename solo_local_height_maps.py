import numpy as np
import matplotlib.pyplot as plt

class SoloLocalHeightMaps:

    def __init__(self):
        # Initialize the height map with zeros (default height)
        self.height_map = np.zeros((33, 21))
        self.first_run = True
        self.pre_trench_call_counter = 0
        self.plotter_counter = 0
        self.obstacle_height = 1.0
        self.k = 0

    def trench(self):

        side_rows = 1
        # Iterate over each point in the grid
        for i in range(self.height_map.shape[0]):  # For each row
            for j in range(self.height_map.shape[1]):  # For each column
                # Determine color based on position (using provided logic)
                if (j % self.height_map.shape[0]) < side_rows:  # Red condition
                    self.height_map[i, j] = self.obstacle_height  # Set height to 1 meter
                elif (j % self.height_map.shape[0]) >= (self.height_map.shape[0]-side_rows):  # Green condition
                    self.height_map[i, j] = self.obstacle_height  # Set height to 1 meter
                # Yellow and Blue conditions are ignored for height assignment
        return self.height_map.flatten()
    
    def pre_trench(self):
        #length = 0.8
        #cell_length = length / self.height_map.shape[0]
        side_rows = 1
        # Number of iterations to complete the transition
        total_shifts = self.height_map.shape[0] - 1
        # Determine how much of the map the obstacles should occupy after 33 iterations
        target_occupation = self.height_map.shape[0] * 0.75
        # Calculate the increment of the obstacle extension per call
        extension_per_call = target_occupation / total_shifts
        # Current extension length based on the number of calls made
        current_extension_length = int(extension_per_call * self.pre_trench_call_counter)

        # Calculate the start index for the obstacles based on the current extension
        start_index_for_obstacle = self.height_map.shape[0] - current_extension_length
        # Iterate over each point in the grid
        for i in range(self.height_map.shape[0]):  # For each row
            for j in range(self.height_map.shape[1]):  # For each column
                # Check if the row index is within the current obstacle range
                if i >= start_index_for_obstacle:
                    if (j % self.height_map.shape[1]) < side_rows:  # Red condition, applied based on row index
                        self.height_map[i, j] = self.obstacle_height
                    elif (j % self.height_map.shape[1]) >= (self.height_map.shape[1] - side_rows):  # Green condition
                        self.height_map[i, j] = self.obstacle_height
                # Yellow and Blue conditions remain ignored for height assignment

        # if self.first_run:
        #     self.plotter_counter += 1
        #     if self.plotter_counter % 5 == 0:
        #         plt.imshow(self.height_map, cmap='viridis', origin='lower', interpolation='none')
        #         plt.colorbar(label='Height (m)')
        #         plt.title('Synthetic Height Map Visualization')
        #         plt.xlabel('X Coordinate')
        #         plt.ylabel('Y Coordinate')
        #         plt.show()
        # Increment the call counter

        if self.k > 1:
            self.k = 0
            self.pre_trench_call_counter += 1
        else:    
            self.k += 1

        if self.pre_trench_call_counter >= total_shifts:
            # print(self.height_map)
            # exit()
            self.trench() # Call trench after fully extended
            return self.height_map.flatten()
        else:
            return self.height_map.flatten()

    #trench()
                    
    def flatbed(self):
        current_height = 0
        # Iterate over each point in the grid
        for i in range(self.height_map.shape[0]):  # For each row
            for j in range(self.height_map.shape[1]):  # For each column
                self.height_map[i, j] = current_height  # Set height
        return self.height_map.flatten()
    #flatbed()
                    
    def starting_ascent(self):
        max_height = 0.4  # Maximum height at the top edge
        length = 0.8  # Total length in meters
        # Calculate the physical size of each cell in the length direction
        cell_length = length / self.height_map.shape[0]
        # Calculate midpoint in the length direction
        midpoint_y = length / 2
        # Iterate over each point in the grid
        for i in range(self.height_map.shape[0]):  # For each row
            # Calculate the physical position of the current cell in the length direction
            y_pos = i * cell_length + cell_length / 2
            # Check if above midpoint (in the top half of the grid)
            if y_pos > midpoint_y:
                # Calculate distance from midpoint in the length direction
                dist_y = abs(midpoint_y - y_pos)
                # Determine the proportional distance from the center to the top edge
                dist_ratio = dist_y / (length / 2)
                # Set height based on distance ratio
                self.height_map[i, :] = dist_ratio * max_height
        return self.height_map.flatten()
    #starting_ascent(height_map)

    def starting_descent(self):
        max_height = 0.4  # Maximum height at the top edge
        length = 0.8  # Total length in meters
        # Calculate the physical size of each cell in the length direction
        cell_length = length / self.height_map.shape[0]
        # Calculate midpoint in the length direction
        midpoint_y = length / 2
        # Iterate over each point in the grid
        for i in range(self.height_map.shape[0]):  # For each row
            # Calculate the physical position of the current cell in the length direction
            y_pos = i * cell_length + cell_length / 2
            if y_pos <= midpoint_y:
                self.height_map[i, :] = max_height
            # Check if above midpoint (in the top half of the grid)
            if y_pos > midpoint_y:
                # Calculate distance from midpoint in the length direction
                dist_y = abs(midpoint_y - y_pos)
                # Determine the proportional distance from the center to the top edge
                dist_ratio = dist_y / (length / 2)
                # Set height based on distance ratio
                self.height_map[i, :] = max_height - (dist_ratio * max_height)
        return self.height_map.flatten()
    
    #starting_descent(self.height_map)

    # Visualize the synthetic height map
    # plt.imshow(self.height_map, cmap='viridis', origin='lower', interpolation='none')
    # plt.colorbar(label='Height (m)')
    # plt.title('Synthetic Height Map Visualization')
    # plt.xlabel('X Coordinate')
    # plt.ylabel('Y Coordinate')
    # plt.show()






# def starting_ascent(self):
#     max_height = 0.4  # Maximum height at the edges
#     width = 0.5  # Total width in meters
#     length = 0.8  # Total length in meters
    
#     # Calculate the physical size of each cell
#     cell_width = width / self.height_map.shape[1]
#     cell_length = length / self.height_map.shape[0]

#     # Calculate midpoint coordinates in meters
#     midpoint_x = width / 2
#     midpoint_y = length / 2

#     # Iterate over each point in the grid
#     for i in range(self.height_map.shape[0]):  # For each row
#         for j in range(self.height_map.shape[1]):  # For each column
#             # Calculate the physical position of the current cell
#             x_pos = j * cell_width + cell_width / 2
#             y_pos = i * cell_length + cell_length / 2
#             # Calculate distance from midpoint in both directions
#             dist_x = abs(midpoint_x - x_pos)
#             dist_y = abs(midpoint_y - y_pos)
            
#             # Determine the proportional distance from the center to the edge
#             dist_ratio = max(dist_x / (width / 2), dist_y / (length / 2))
            
#             # Set height based on distance ratio
#             self.height_map[i, j] = dist_ratio * max_height

# starting_ascent(self.height_map)