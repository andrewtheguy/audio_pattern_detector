
import numpy as np
def find_nearest_distance(array, value):
    array = np.asarray(array)
    arr2=(value - array)
    idx = arr2[arr2 >= 0].argmin()
    return arr2[idx]

print(find_nearest_distance([1,2,3,9,10,11],3.5)) 