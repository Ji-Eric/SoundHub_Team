import os
# Force TensorFlow to disable Metal GPU support (run on CPU)
os.environ['TF_MTL_DISABLE'] = '1'  # disable Metal plugin
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # also hide any GPU devices

import tensorflow as tf
# Alternatively, explicitly set visible devices to an empty list (CPU-only)
tf.config.set_visible_devices([], 'GPU')

import os
import glob
import random
import numpy as np
import pandas as pd
import torch
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

from sklearn.model_selection import StratifiedKFold
import bioacoustics_model_zoo as bmz  # import model zoo as bmz
from opensoundscape.ml.shallow_classifier import quick_fit
from matplotlib import pyplot as plt

plt.rcParams['figure.figsize'] = [15, 5]

torch.manual_seed(0)
random.seed(0)
np.random.seed(0)

#adj to local proj structure
base_dir = os.path.join(os.getcwd(), "data", "non-avian_ML", "audio")
print("Base directory:", base_dir)

# Get all species subfolders
species_list = [
    d for d in os.listdir(base_dir)
    if os.path.isdir(os.path.join(base_dir, d))
]
print("Species found:", species_list)

#loop & build labels

all_species_dfs = {}

for species in species_list:
    data_5s_path = os.path.join(base_dir, species, "data_5s")
    if not os.path.exists(data_5s_path):
        print(f"Skipping {species}: no 5-second data found.")
        continue

    pos_dir = os.path.join(data_5s_path, "pos")
    neg_dir = os.path.join(data_5s_path, "neg")

    # Gather .wav files recursively
    pos_files = glob.glob(os.path.join(pos_dir, "**", "*.wav"), recursive=True)
    neg_files = glob.glob(os.path.join(neg_dir, "**", "*.wav"), recursive=True)

    # Build a DataFrame with file paths and binary labels (1 for positive, 0 for negative)
    pos_df = pd.DataFrame({"file": pos_files, "present": 1})
    neg_df = pd.DataFrame({"file": neg_files, "present": 0})
    species_df = pd.concat([pos_df, neg_df], ignore_index=True)
    species_df["species"] = species  # add species info
    species_df["file"] = species_df["file"].astype(str)
    species_df.set_index("file", inplace=True)

    all_species_dfs[species] = species_df
    print(f"Found {len(species_df)} total clips for {species} (5s).")

#strat k fold for each species

#5 slpits
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for species, df in all_species_dfs.items():
    print(f"\n===== {species.upper()} =====")

    #file data paths as indexes and present as y values
    X = df.index.values
    y = df["present"].values

    fold_num = 1
    for train_idx, val_idx in skf.split(X, y):
        print(f"\n--- Fold {fold_num} ---")
        fold_num += 1

        train_files = X[train_idx]
        val_files = X[val_idx]

        train_labels = df.loc[train_files]
        val_labels = df.loc[val_files]

        print(f"Train set size: {len(train_labels)} | Validation set size: {len(val_labels)}")

       #use perch model
        model = bmz.Perch()
        model.change_classes(['present'])

        # Generate embeddings for training and validation sets
        emb_train = model.embed(train_labels, return_dfs=False, batch_size=128, num_workers=0)
        emb_val = model.embed(val_labels, return_dfs=False, batch_size=128, num_workers=0)

        # Train a shallow classifier on the embeddings using quick_fit
        quick_fit(
            model.network,
            emb_train,
            train_labels['present'].values,
            emb_val,
            val_labels['present'].values,
            steps=1000
        )

        #use validation set for predicitons
        predictions = model.predict(val_labels.index)
        print("Predictions:", predictions)

        #histograms
        plt.figure()
        plt.hist(predictions[val_labels['present'] == 1], bins=20, alpha=0.5, label='Positives')
        plt.hist(predictions[val_labels['present'] == 0], bins=20, alpha=0.5, label='Negatives')
        plt.legend()
        plt.title(f"{species.upper()} - Fold {fold_num-1} Predictions")
        plt.show()

print("\ndone!!!.")
