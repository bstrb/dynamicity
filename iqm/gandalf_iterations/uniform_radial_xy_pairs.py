import math
import matplotlib.pyplot as plt

def grid_points_in_circle(x_center, y_center, max_radius, step=0.5):
    """
    Generate all grid points inside a circle with the given center and maximum radius.
    The grid is defined by the specified step size (granularity) and the coordinates are 
    rounded to a number of decimals determined by the step size.
    
    Args:
        x_center, y_center: Coordinates of the circle center.
        max_radius: Maximum radius from the center.
        step: Grid spacing.
        
    Returns:
        A list of (x, y) tuples that lie within the circle.
    """
    # Determine the number of decimals for rounding based on the step size.
    decimals = max(0, -int(math.floor(math.log10(step))))
    
    points = []
    max_i = int(math.ceil(max_radius / step))
    
    for i in range(-max_i, max_i + 1):
        for j in range(-max_i, max_i + 1):
            # Compute and round the grid coordinates
            x = round(x_center + i * step, decimals)
            y = round(y_center + j * step, decimals)
            # Check if the point is within the circle
            if (x - x_center) ** 2 + (y - y_center) ** 2 <= max_radius ** 2:
                points.append((x, y))
    return points

def generate_sorted_grid_points(max_radius, step=0.5):
    """
    Generate all grid points (with the given granularity) within a circle defined by max_radius,
    round them based on the step size, and sort them in order of increasing radial distance from the center.
    
    Returns:
        List of (x, y) tuples sorted from the center outward.
    """
    x_center, y_center = 0,0
    points = grid_points_in_circle(x_center, y_center, max_radius, step)
    # Sort by the squared distance from the center (no need for square roots)
    points.sort(key=lambda pt: (pt[0] - x_center) ** 2 + (pt[1] - y_center) ** 2)
    print(f"Generated {len(points)} grid points in the circle.")
    return points

# Example usage:
if __name__ == "__main__":
    center_x, center_y = 0, 0
    max_radius = 1    # maximum radius of the circle
    step = 0.5        # grid granularity
    
    # points = grid_points_in_circle(center_x, center_y, max_radius, step)
    points = generate_sorted_grid_points(max_radius, step)
    
    # Quick visual check
    xs, ys = zip(*points)
    plt.scatter(xs, ys)
    plt.gca().set_aspect('equal', 'box')
    plt.title("Grid Points in a Circle Sorted by Radial Distance (Rounded Coordinates)")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.show()
