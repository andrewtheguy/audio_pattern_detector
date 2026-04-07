def max_distance(sorted_data: list[float]) -> float:
    """Find the maximum distance between consecutive elements in sorted data."""
    max_dist: float = 0
    for i in range(1, len(sorted_data)):
        dist = sorted_data[i] - sorted_data[i - 1]
        max_dist = max(max_dist, dist)
    return max_dist
