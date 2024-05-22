
import numpy as np

from utils import find_nearest_distance_backwards, find_nearest_distance_forward


print(find_nearest_distance_backwards([1,2,3,9,10,11],3.5)) 
print(find_nearest_distance_backwards([1,2,3,10,7.5,9,10,11],8)) 
print(find_nearest_distance_forward([1,2,3,9,10,11],3.5)) 