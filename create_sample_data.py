import csv
import os
import numpy as np

# Ensure data directory exists
os.makedirs('data', exist_ok=True)
csv_file = 'data/dataset.csv'

# Define gesture labels to generate sample data for
labels = ['A', 'B', 'Hello', 'Thanks']
samples_per_label = 50  # Number of dummy samples per gesture

# MediaPipe Hand Landmarks: 21 landmarks with normalized (x, y) coordinates
# 21 x-coordinates + 21 y-coordinates = 42 features
header = ['label'] + [f'x{i}' for i in range(21)] + [f'y{i}' for i in range(21)]

np.random.seed(42)  # For reproducible random numbers

with open(csv_file, mode='w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(header)

    for idx, label in enumerate(labels):
        for _ in range(samples_per_label):
            # Generate distinct feature patterns per class with small random noise
            # Base values are shifted per class (idx * 0.25) so the model can learn clear boundaries
            base_pattern = np.linspace(-0.5, 0.5, 42) + (idx * 0.25)
            noise = np.random.normal(0, 0.05, 42)
            features = (base_pattern + noise).tolist()
            
            writer.writerow([label] + features)

print(f"Sample dataset successfully created at '{csv_file}'!")
print(f"Generated {len(labels) * samples_per_label} total rows for labels: {labels}")