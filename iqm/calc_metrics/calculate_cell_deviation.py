import numpy as np

# Function to calculate normalized deviation from target cell parameters
def calculate_cell_deviation(cell_params, target_params):
    # Extract cell parameters
    a, b, c, al, be, ga = cell_params
    target_a, target_b, target_c, target_al, target_be, target_ga = target_params

    # Normalize deviations by dividing by target values to make them dimensionless
    length_deviation = np.sqrt((a - target_a) ** 2 + (b - target_b) ** 2 + (c - target_c) ** 2)
    angle_deviation = np.sqrt((al - target_al) ** 2 + (be - target_be) ** 2 + (ga - target_ga) ** 2)
    
    return length_deviation, angle_deviation
