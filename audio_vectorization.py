# audio_vectorization_pipeline.py

import os
import numpy as np
import pandas as pd
import librosa
from tensorflow.keras.preprocessing.sequence import pad_sequences

# ----------------------------
# PARAMETERS
# ----------------------------
BASE_DIR = "audio_speech"
LABELS_CSV = "ravdess_labels.csv"
N_MFCC = 13
SAVE_TO_DISK = True

# ----------------------------
# LOAD LABELS
# ----------------------------
labels_df = pd.read_csv(LABELS_CSV)

# ----------------------------
# INITIALIZE STORAGE
# ----------------------------
mfcc_features = []   # Classical ML: fixed-length vectors
mfcc_sequences = []  # Deep Learning: sequences
file_names = []      # Track file order

# ----------------------------
# FEATURE EXTRACTION
# ----------------------------
for actor in os.listdir(BASE_DIR):
    actor_path = os.path.join(BASE_DIR, actor)
    if os.path.isdir(actor_path):
        for file in os.listdir(actor_path):
            if file.endswith(".wav"):
                file_path = os.path.join(actor_path, file)
                y, sr = librosa.load(file_path, sr=None)

                # Extract MFCCs
                mfcc = librosa.feature.mfcc(
                    y=y, 
                    sr=sr, 
                    n_mfcc=N_MFCC,
                    n_fft=int(0.025*sr),    # 25ms window
                    hop_length=int(0.01*sr) # 10ms step
                    )

                # Classical ML: average across time
                mfcc_mean = np.mean(mfcc, axis=1)
                mfcc_features.append(mfcc_mean)

                # Deep Learning: keep full sequence (time_steps x n_mfcc)
                mfcc_sequences.append(mfcc.T)

                # Keep track of filename
                file_names.append(file)

# ----------------------------
# CLASSICAL ML DATASET
# ----------------------------
mfcc_df = pd.DataFrame(mfcc_features, columns=[f"mfcc_{i+1}" for i in range(N_MFCC)])
mfcc_df.insert(0, "file_name", file_names)

# Merge labels
classical_df = pd.merge(mfcc_df, labels_df, on="file_name")

# Features (X) and labels (y) for classical ML
X_classical = classical_df[[f"mfcc_{i+1}" for i in range(N_MFCC)]].values
y_classical = classical_df["Emotion"].values  # already int32

# ----------------------------
# DEEP LEARNING DATASET
# ----------------------------
# Pad sequences to same length
X_deep = pad_sequences(mfcc_sequences, padding='post', dtype='float32')

# Align labels with file_names order
deep_labels_df = labels_df.set_index('file_name').loc[file_names].reset_index()
y_deep = deep_labels_df["Emotion"].values  # already int32

# ----------------------------
# OUTPUT INFO
# ----------------------------
print("Classical ML dataset shape:", X_classical.shape)
print("Deep Learning dataset shape:", X_deep.shape)
print("Number of unique labels:", len(np.unique(y_classical)))
print("Labels range:", np.min(y_classical), "to", np.max(y_classical))

# ----------------------------
# SAVE DATASETS TO DISK
# ----------------------------
if SAVE_TO_DISK:
    np.save("X_classical.npy", X_classical)
    np.save("y_classical.npy", y_classical)
    np.save("X_deep.npy", X_deep)
    np.save("y_deep.npy", y_deep)
    print("Datasets saved to disk.")