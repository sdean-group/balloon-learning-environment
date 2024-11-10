import numpy as np

import matplotlib.pyplot as plt

# Load the data from the .npy file
data = np.load('/height_displacement.npy')
# data = np.load('action_list.npy')

# Create a line plot
plt.plot(data[:20])
plt.title('Height Displacement')
plt.xlabel('Index')
plt.ylabel('Displacement')
plt.grid(True)

# Show the plot
plt.show()