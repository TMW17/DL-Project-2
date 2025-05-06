import numpy as np
import pandas as pd

# Parameters
num_samples = 100  # Number of synthetic ECG beats
window_size = 187  # Number of samples per beat
num_classes = 5    # Number of label classes (N, S, V, F, Q)

# Generate synthetic ECG signals
# Simulate simple sinusoidal patterns with noise to mimic ECG beats
np.random.seed(42)
X = []
for _ in range(num_samples):
    t = np.linspace(0, 2 * np.pi, window_size)
    # Create a synthetic ECG-like signal with peaks and noise
    signal = 0.5 * np.sin(t) + 0.3 * np.sin(2 * t) + 0.1 * np.random.normal(0, 0.05, window_size)
    X.append(signal)

X = np.array(X)

# Generate random labels (0 to 4)
y = np.random.randint(0, num_classes, num_samples)

# Combine features and labels
data = np.hstack((X, y.reshape(-1, 1)))

# Create DataFrame without header
df = pd.DataFrame(data)

# Save to CSV without header
df.to_csv('synthetic_mitbih_test.csv', index=False, header=False)

print(f"Synthetic test CSV 'synthetic_mitbih_test.csv' generated with {num_samples} samples.")