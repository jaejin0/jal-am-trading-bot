import numpy as np

data = [[0, 0, 0, 0],
        [1, 1, 1, 1],
        [2, 2, 2, 2],
        [3, 3, 3, 3],
        [4, 4, 4, 4],
        [5, 5, 5, 5]]

data = np.array(data)

timestep = 4
time_range = 2

print(data[timestep: : -1])
