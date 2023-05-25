import numpy as np
import seaborn as sns

def populate_hourglass(arr, initial_point):
    rows, cols = arr.shape
    row, col = initial_point
    
    if rows < 5 or cols < 5:
        raise ValueError("Array dimensions must be at least 5x5.")

    if row < 0 or row + 4 >= rows or col < 0 or col + 4 >= cols:
        raise ValueError("Initial point is out of bounds.")
    
    arr[row+2, col] = 1  # Top of the hourglass
    arr[row+2, col+1] = 1  # Top of the hourglass
    arr[row+2, col+2] = 1  # Top of the hourglass
    arr[row+1, col+1] = 1  # Middle of the hourglass
    arr[row+3, col+1] = 1  # Middle of the hourglass
    arr[row, col+2] = 1  # Bottom of the hourglass
    arr[row+4, col+2] = 1  # Bottom of the hourglass
    arr[row+1, col+3] = 1  # Bottom of the hourglass
    arr[row+3, col+3] = 1  # Bottom of the hourglass
    arr[row+2, col+4] = 1  # Bottom of the hourglass
    
    return arr

# Create a zeros ndarray with shape (25, 25)
arr = np.zeros((25, 25))

# Specify the initial point
initial_point = (10, 10)

# Populate the ndarray with an hourglass shape
result = populate_hourglass(arr, initial_point)

# Print the resulting ndarray
print(result)
sns.heatmap(result)