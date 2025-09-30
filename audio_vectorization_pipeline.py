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
EXTRACT_TEST_SET = False    # whether test set is needed
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
    "val": val_files,
    "test": test_files
}

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

        # Classical ML: average across time
        mfcc_mean = np.mean(mfcc, axis=1)
        mfcc_features.append(mfcc_mean)

        # Deep Learning: keep full sequence
        mfcc_sequences.append(mfcc.T)

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
# EXTRACT FEATURES FOR SPLITS
# ----------------------------

# If test set is not needed yet, remove it from the output so we don't have to extract features
if not EXTRACT_TEST_SET and "test" in splits:
    del splits["test"]

datasets = {}

for split_name, files in splits.items():
    feature_start = time.time()

    print(f"\nExtracting features for {split_name} set...")
    X_classical, y_classical, X_deep, y_deep = extract_features(files)
    datasets[split_name] = {
        "X_classical": X_classical,
        "y_classical": y_classical,
        "X_deep": X_deep,
        "y_deep": y_deep
    }

    # Print time elapsed
    feature_end = time.time()
    feature_elapsed = feature_end - feature_start
    print(f"Feature extraction for {split_name} set took {feature_elapsed:.2f} seconds")


# ----------------------------
# OUTPUT INFO
# ----------------------------
for split, data in datasets.items():
    print(f"\n{split.upper()} SET")
    print("  Classical ML:", data["X_classical"].shape, data["y_classical"].shape)
    print("  Deep Learning:", data["X_deep"].shape, data["y_deep"].shape)

# ----------------------------
# SAVE DATASETS TO DISK
# ----------------------------
if SAVE_TO_DISK:
    for split, data in datasets.items():
        np.save(f"X_classical_{split}.npy", data["X_classical"])
        np.save(f"y_classical_{split}.npy", data["y_classical"])
        np.save(f"X_deep_{split}.npy", data["X_deep"])
        np.save(f"y_deep_{split}.npy", data["y_deep"])
    print("\nDatasets saved to disk.")

# ----------------------------
# PRINT PIPELINE TIME ELAPSED
# ----------------------------
end_time = time.time()
elapsed = end_time - start_time
print(f"\nPipeline completed in {elapsed:.2f} seconds")