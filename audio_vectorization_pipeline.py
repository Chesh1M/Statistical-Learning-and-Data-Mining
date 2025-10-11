import os
import numpy as np
import pandas as pd
import librosa
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
import time

# ----------------------------
# PARAMETERS
# ----------------------------
BASE_DIR = "audio_speech"
LABELS_CSV = "ravdess_labels.csv"
N_MFCC = 13
SAVE_TO_DISK = True
TEST_SIZE = 0.15   # 15% test
VAL_SIZE = 0.15    # 15% validation (relative to total)
WINDOW_SIZE = 25/1000   # 25ms (change to test different settings)
STEP_SIZE = 10/1000     # 10ms (change as test different settings)
EXTRACT_TEST_SET = True    # whether test set is needed
SEED = 42

start_time = time.time()

# ----------------------------
# LOAD LABELS
# ----------------------------
labels_df = pd.read_csv(LABELS_CSV)

# ----------------------------
# TRAIN/VAL/TEST SPLIT (file-level)
# ----------------------------
all_files = labels_df["file_name"].values

# First split train+val vs test
train_val_files, test_files = train_test_split(
    all_files, test_size=TEST_SIZE, random_state=SEED,
    stratify=labels_df["Emotion"].values
)

# Then split train vs val
train_files, val_files = train_test_split(
    train_val_files, test_size=VAL_SIZE/(1-TEST_SIZE), random_state=SEED,
    stratify=labels_df.set_index("file_name").loc[train_val_files]["Emotion"].values
)

splits = {
    "train": train_files,
    "val": val_files
}

if EXTRACT_TEST_SET:
    splits["test"] = test_files

print(f"Split sizes: train={len(train_files)}, val={len(val_files)}, test={len(test_files)}")

# ----------------------------
# FEATURE EXTRACTION FUNCTION
# ----------------------------
def extract_features(file_list, base_dir=BASE_DIR, n_mfcc=N_MFCC):
    mfcc_features = []   # Classical ML
    mfcc_sequences = []  # Deep Learning
    labels = []

    for file in file_list:
        # Find actor folder
        actor = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d)) and file in os.listdir(os.path.join(base_dir, d))]
        if not actor:
            continue
        file_path = os.path.join(base_dir, actor[0], file)

        # Load audio
        y, sr = librosa.load(file_path, sr=None)

        # Extract MFCCs
        mfcc = librosa.feature.mfcc(
            y=y, sr=sr, n_mfcc=n_mfcc,
            n_fft=int(WINDOW_SIZE*sr),    # 25ms window 
            hop_length=int(STEP_SIZE*sr) # 10ms step 
        )

        # Compute delta and double-delta 
        delta = librosa.feature.delta(mfcc)
        delta2 = librosa.feature.delta(mfcc, order=2)

        # Stack all features
        mfcc_combined = np.vstack([mfcc, delta, delta2])

        # Classical ML: average across time
        mfcc_mean = np.mean(mfcc_combined, axis=1)
        mfcc_features.append(mfcc_mean)

        # Deep Learning: keep full sequence
        mfcc_sequences.append(mfcc_combined.T)

        # Label
        emotion = labels_df.set_index("file_name").loc[file, "Emotion"]
        labels.append(emotion)

    # Convert to arrays
    X_classical = np.array(mfcc_features)
    y_classical = np.array(labels)

    X_deep = pad_sequences(mfcc_sequences, padding='post', dtype='float32')
    y_deep = np.array(labels)

    return X_classical, y_classical, X_deep, y_deep

# ----------------------------
# EXPORT FUNCTIONS
# ----------------------------
def export_classical_to_csv(X, y, split_name):
    # Define column / feature names
    feature_names = (
        [f"mfcc_{i+1}" for i in range(N_MFCC)] + 
        [f"delta_{i+1}" for i in range(N_MFCC)] + 
        [f"delta2_{i+1}" for i in range(N_MFCC)]
    )
    # Convert array into dataframe
    df = pd.DataFrame(X, columns=feature_names)
    # Define the target / output column
    df["Emotion"] = y
    # Export as csv
    df.to_csv(f"classical_{split_name}.csv", index=False)
    print(f"✅ Saved classical features: classical_{split_name}.csv — shape {df.shape}")

def export_deep_to_npy(X, y, split_name):
    # Export directly as npy file
    np.save(f"X_deep_{split_name}.npy", X)
    np.save(f"y_deep_{split_name}.npy", y)
    print(f"✅ Saved deep features: X_deep_{split_name}.npy ({X.shape}), y_deep_{split_name}.npy ({y.shape})")

# ----------------------------
# MAIN EXTRACTION LOOP
# ----------------------------
for split_name, files in splits.items():
    feature_start = time.time()
    print(f"\n🔸 Extracting features for {split_name} set...")

    X_classical, y_classical, X_deep, y_deep = extract_features(files)

    # Export datasets for Classical ML (CSV)
    export_classical_to_csv(X_classical, y_classical, split_name)

    # Export datasets for Deep Learning (npy)
    export_deep_to_npy(X_deep, y_deep, split_name)

    print(f"⏱️ {split_name} extraction completed in {time.time() - feature_start:.2f}s")

# ----------------------------
# PRINT PIPELINE TIME ELAPSED
# ----------------------------
end_time = time.time()
elapsed = end_time - start_time
print(f"\nPipeline completed in {elapsed:.2f} seconds")