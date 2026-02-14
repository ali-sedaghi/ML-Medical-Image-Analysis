import os
import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from datastats import extract_patient_stats
from distribution import get_class_distribution
from loader import load_patient_data
from show import show_slice

# Set dark style
plt.style.use("dark_background")

# BraTS2020 uploads on Kaggle
DATA_PATH = "../input/datasets/awsaf49/brats20-dataset-training-validation/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData"

# Load the mapping CSV
csv_path = "/kaggle/input/datasets/awsaf49/brats20-dataset-training-validation/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData/name_mapping.csv"
survival_path = "/kaggle/input/datasets/awsaf49/brats20-dataset-training-validation/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData/survival_info.csv"

# Get list of train directories
train_dirs = [f.path for f in os.scandir(DATA_PATH) if f.is_dir()]
print(f"Total Patients found: {len(train_dirs)}")


# Pick a random patient
random_patient_path = random.choice(train_dirs)
print(f"Visualizing Patient: {os.path.basename(random_patient_path)}")

# Load data
t1, t1ce, t2, flair, seg = load_patient_data(random_patient_path)

# Pick a slice that actually has a tumor (we scan for one)
# We sum the segmentation mask to find a slice with high activity
slice_with_max_tumor = np.argmax(np.sum(seg, axis=(0, 1)))
print(f"Showing Slice: {slice_with_max_tumor}")
show_slice(t1, t1ce, t2, flair, seg, slice_with_max_tumor)


# Run stats on a subset
counts = get_class_distribution(train_dirs, num_samples=50)

# Convert to DataFrame for Plotting
df_counts = pd.DataFrame(list(counts.items()), columns=["Class", "Count"])
# Map Class IDs to Names
class_map = {0: "Background", 1: "NCR/NET", 2: "Edema", 4: "Enhancing Tumor"}
df_counts["Label"] = df_counts["Class"].map(class_map)

# Exclude Background for better scale visualization of tumor classes
df_tumor = df_counts[df_counts["Class"] != 0]

# Plot
plt.figure(figsize=(12, 6))
sns.barplot(
    data=df_tumor, x="Label", y="Count", hue="Label", palette="viridis", legend=False
)
plt.title("Tumor Class Distribution (Excluding Background)")
plt.ylabel("Pixel Count")
plt.show()


df_survival = pd.read_csv(survival_path)
print("Survival Data Loaded:")
display(df_survival.head())

plt.figure(figsize=(10, 5))
sns.histplot(df_survival["Age"], kde=True, bins=20, color="teal")
plt.title("Distribution of Patient Ages")
plt.show()


df_stats = extract_patient_stats(train_dirs, num_samples=50)

# Print Shape and Values for the first few samples
print("\n--- Individual Patient Statistics (First 10) ---")
display(df_stats.head(10))

# Check for inconsistent shapes
unique_shapes = df_stats["Shape"].unique()
print(f"\nUnique Shapes found in dataset: {unique_shapes}")
if len(unique_shapes) > 1:
    print("WARNING: Dataset contains variable volume dimensions!")
else:
    print("PASS: All volumes have consistent dimensions.")

# Global Mean Statistics (Average across the dataset)
print("\n--- Global Mean Values per Channel ---")
display(df_stats[["T1_mean", "T1ce_mean", "T2_mean", "FLAIR_mean"]].describe())

# Visualize the spread of means
plt.figure(figsize=(12, 6))
sns.boxplot(data=df_stats[["T1_mean", "T1ce_mean", "T2_mean", "FLAIR_mean"]])
plt.title("Distribution of Mean Intensity Values per Channel")
plt.ylabel("Mean Pixel Intensity")
plt.show()
