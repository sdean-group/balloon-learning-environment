import json
import numpy as np

# Load the JSON file
file_path = "eval/test/mpc.json"  # Replace with your actual file path

with open(file_path, 'r') as file:
    data = json.load(file)

data = data[0]
print(data['seed'])

# Extract power values from the flight path
power_values = [point["power"] for point in data["flight_path"]]

# Convert to a NumPy array
power_array = np.array(power_values)

print(np.min(power_array))

# print("Power Values Array:", power_array)