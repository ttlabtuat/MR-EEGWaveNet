import numpy as np
from sklearn.preprocessing import StandardScaler
from pyod.models.ecod import ECOD
import matplotlib.pyplot as plt

# Set dimensions
S = 30  # Number of samples
C = 19  # Number of channels
N = 1000  # Number of time points

# Generate random data with the given dimensions
epoch_data = np.random.rand(S, C, N)
print(epoch_data.shape)  # Should output (30, 19, 1000)

# Reshape the 3D epoch_data into 2D array (samples, features)
# Flatten each sample's channels and time points into a single feature vector
Xt = epoch_data.reshape(epoch_data.shape[0], -1)

# Initialize a scaler to standardize features to zero mean and unit variance
scaler = StandardScaler()

# Fit the scaler on data and transform it
Xt = scaler.fit_transform(Xt)

# Fit ECOD anomaly detection model
clf = ECOD() 

# Train the model on the scaled data
clf.fit(Xt)


# -----------------------------
# Step : Calculate anomaly scores and threshold
# -----------------------------
# Compute anomaly scores for each sample
# Normalize by the number of features to scale the score per feature
Anomaly_score = clf.decision_function(Xt)/Xt.shape[1]

# Set threshold as the mean anomaly score across all samples
threshold = np.mean(Anomaly_score)
print(f"threshold: {threshold}")

# -----------------------------
# Step : Create base prediction vector
# -----------------------------
p_ones = 0.25 # proportion of 1's

# Number of ones based on desired proportion
num_ones = round(S * p_ones)

# Create vector with the correct number of ones and zeros
pred_base = np.array([1] * num_ones + [0] * (S - num_ones))

# Shuffle so that ones and zeros are randomly positioned
np.random.shuffle(pred_base)

# -----------------------------
# Step : Post-process based on anomaly score
# -----------------------------
# Make a copy to preserve the original base vector
pred_post = pred_base.copy()

# Set predictions to 0 if the anomaly score is below the threshold
pred_post[Anomaly_score < threshold] = 0

# -----------------------------
# Step : Output
# -----------------------------
print(f"pred_base: {pred_base}")
print(f"Proportion of 1's: {np.mean(pred_base):.2f}")

print(f"pred_post: {pred_post}")
print(f"Proportion of 1's: {np.mean(pred_post):.2f}")

