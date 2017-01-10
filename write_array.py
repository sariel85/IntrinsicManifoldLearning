import numpy as np

# Generate some test data
data = np.arange(200).reshape((4,5,10))

# Write the array to disk
with file('test.txt', 'w') as outfile:
    for data_slice in data:
        np.savetxt(outfile, data_slice, fmt='%-7.2f')