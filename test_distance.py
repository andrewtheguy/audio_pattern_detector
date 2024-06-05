import math


# Function to calculate Euclidean distance between two points
def point_distance(x1, y1, x2, y2):
    return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)


# Function to calculate the distance from point (px, py) to line segment (x1, y1)-(x2, y2)
def point_to_segment_distance(px, py, x1, y1, x2, y2):
    line_mag = point_distance(x1, y1, x2, y2)
    if line_mag == 0:
        return point_distance(px, py, x1, y1)
    u = ((px - x1) * (x2 - x1) + (py - y1) * (y2 - y1)) / (line_mag ** 2)
    if u < 0:
        return point_distance(px, py, x1, y1)
    elif u > 0 and u < 1:
        x_closest = x1 + u * (x2 - x1)
        y_closest = y1 + u * (y2 - y1)
        return point_distance(px, py, x_closest, y_closest)
    else:
        return point_distance(px, py, x2, y2)


# Function to calculate the minimum distance between two piecewise linear curves
def calculate_min_distance(y1, y2):
    n = len(y1)
    if n != len(y2):
        raise ValueError("Arrays y1 and y2 must have the same length.")

    min_distance = float('inf')

    # Iterate through each segment pair
    for i in range(n - 1):
        for j in range(n - 1):
            # Calculate distances between segment endpoints
            dist1 = point_to_segment_distance(i, y1[i], j, y2[j], j + 1, y2[j + 1])
            dist2 = point_to_segment_distance(i + 1, y1[i + 1], j, y2[j], j + 1, y2[j + 1])
            dist3 = point_to_segment_distance(j, y2[j], i, y1[i], i + 1, y1[i + 1])
            dist4 = point_to_segment_distance(j + 1, y2[j + 1], i, y1[i], i + 1, y1[i + 1])
            dist = min(dist1, dist2, dist3, dist4)
            min_distance = min(min_distance, dist)

    return min_distance


# Example arrays (representing bell-shaped curves)
y1 = [1, 3, 5, 7, 9]
y2 = [2, 4, 6, 8, 10]

min_distance = calculate_min_distance(y1, y2)
print("Minimum Distance:", min_distance)