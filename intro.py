# BirdCLEF+ 2026
# EfficientNetV2-B2 on log-mel spectrograms (TensorFlow / Keras)

# INSTALL & IMPORTS
!pip install -q librosa colorednoise

import os
import gc
import math
import random
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import librosa
import colorednoise as cn
from sklearn.model_selection import StratifiedKFold

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import mixed_precision

warnings.filterwarnings("ignore")

print(f"TensorFlow  : {tf.__version__}")
print(f"GPUs found  : {tf.config.list_physical_devices('GPU')}")

# CONFIG
class CFG:
    BASE_DIR               = Path("/kaggle/input/competitions/birdclef-2026/")
    TRAIN_AUDIO_DIR        = BASE_DIR / "train_audio"
    TRAIN_SOUNDSCAPES_DIR  = BASE_DIR / "train_soundscapes"
    TEST_SOUNDSCAPES_DIR   = BASE_DIR / "test_soundscapes"
    TRAIN_CSV              = BASE_DIR / "train.csv"
    TAXONOMY_CSV           = BASE_DIR / "taxonomy.csv"
    SOUNDSCAPE_LABELS_CSV  = BASE_DIR / "train_soundscapes_labels.csv"
    SAMPLE_SUB_CSV         = BASE_DIR / "sample_submission.csv"

    SAMPLE_RATE = 32000
    DURATION    = 5
    N_SAMPLES   = SAMPLE_RATE * DURATION

    N_FFT      = 1024
    HOP_LENGTH = 320
    N_MELS     = 128
    FMIN       = 20
    FMAX       = 16000

    IMG_HEIGHT = 128
    IMG_WIDTH  = 384
    IMG_SHAPE  = (IMG_HEIGHT, IMG_WIDTH, 3)

    BACKBONE   = "EfficientNetV2B2"
    DROP_RATE  = 0.3
    DROP_RATE2 = 0.2

    BATCH_SIZE    = 32
    EPOCHS        = 3
    LEARNING_RATE = 1e-3
    MIN_LR        = 1e-6
    WEIGHT_DECAY  = 1e-4
    WARMUP_EPOCHS = 3
    PATIENCE      = 7
    N_FOLDS       = 5
    TRAIN_FOLDS   = [0]
    SEED          = 42

    USE_MIXUP        = True
    MIXUP_ALPHA      = 0.4
    USE_SPEC_AUGMENT = True
    FREQ_MASK_MAX    = 20
    TIME_MASK_MAX    = 40
    USE_NOISE        = True
    NOISE_PROB       = 0.5
    SNR_DB_MIN       = 5
    SNR_DB_MAX       = 30

    SECONDARY_LABEL_WEIGHT = 0.5
    MIN_RATING = 0.0

    USE_MIXED_PRECISION = True

    SOUNDSCAPE_DURATION = 60
    INFER_OVERLAP       = 0.0
    INFER_BATCH_SIZE    = 64


def set_seed(seed=CFG.SEED):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

set_seed()

if CFG.USE_MIXED_PRECISION:
    mixed_precision.set_global_policy("mixed_float16")
    print("Mixed precision: ON")

# DATA LOADING
train_df   = pd.read_csv(CFG.TRAIN_CSV)
taxonomy   = pd.read_csv(CFG.TAXONOMY_CSV)
sample_sub = pd.read_csv(CFG.SAMPLE_SUB_CSV)
sc_labels  = pd.read_csv(CFG.SOUNDSCAPE_LABELS_CSV)

ALL_SPECIES    = [c for c in sample_sub.columns if c != "row_id"]
SPECIES_TO_IDX = {s: i for i, s in enumerate(ALL_SPECIES)}
NUM_CLASSES    = len(ALL_SPECIES)

print(f"Total species: {NUM_CLASSES}")
print(f"Training files: {len(train_df)}")

train_df["filepath"] = train_df["filename"].apply(
    lambda x: str(CFG.TRAIN_AUDIO_DIR / x)
)

if CFG.MIN_RATING > 0:
    train_df = train_df[
        (train_df["rating"] >= CFG.MIN_RATING) | (train_df["rating"] == 0)
    ].reset_index(drop=True)

train_df = train_df[
    train_df["primary_label"].isin(SPECIES_TO_IDX)
].reset_index(drop=True)

train_df["label_idx"] = train_df["primary_label"].map(SPECIES_TO_IDX)

print(f"Filtered samples: {len(train_df)}")

# CROSS VALIDATION
skf = StratifiedKFold(n_splits=CFG.N_FOLDS, shuffle=True, random_state=CFG.SEED)
train_df["fold"] = -1

for fold, (_, val_idx) in enumerate(skf.split(train_df, train_df["label_idx"])):
    train_df.loc[val_idx, "fold"] = fold

print(train_df["fold"].value_counts().sort_index())

# AUDIO PREPROCESSING
def _make_mel_filterbank():
    return tf.signal.linear_to_mel_weight_matrix(
        num_mel_bins=CFG.N_MELS,
        num_spectrogram_bins=CFG.N_FFT // 2 + 1,
        sample_rate=CFG.SAMPLE_RATE,
        lower_edge_hertz=CFG.FMIN,
        upper_edge_hertz=CFG.FMAX,
        dtype=tf.float32,
    )

MEL_FILTERBANK = _make_mel_filterbank()

def load_audio(filepath, sr=CFG.SAMPLE_RATE, offset=0.0, duration=None):
    try:
        audio, _ = librosa.load(filepath, sr=sr, offset=offset,
                                duration=duration, mono=True)
        return audio.astype(np.float32)
    except:
        return np.zeros(CFG.N_SAMPLES, dtype=np.float32)

# MODEL
def build_model(num_classes=NUM_CLASSES):
    inputs = keras.Input(shape=CFG.IMG_SHAPE)

    backbone = tf.keras.applications.EfficientNetV2B2(
        include_top=False,
        weights="imagenet",
        input_shape=CFG.IMG_SHAPE,
    )
    backbone.trainable = True

    x = backbone(inputs, training=True)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(CFG.DROP_RATE)(x)
    x = layers.Dense(512)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("swish")(x)
    x = layers.Dropout(CFG.DROP_RATE2)(x)

    outputs = layers.Dense(
        num_classes,
        activation="sigmoid",
        dtype="float32",
    )(x)

    return keras.Model(inputs, outputs)

# LOSS
def focal_loss(gamma=2.0, alpha=0.25):
    def _loss(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1 - 1e-7)

        bce = -(y_true * tf.math.log(y_pred) +
                (1 - y_true) * tf.math.log(1 - y_pred))
        p_t = y_true * y_pred + (1 - y_true) * (1 - y_pred)
        a_t = y_true * alpha + (1 - y_true) * (1 - alpha)
        return tf.reduce_mean(a_t * tf.pow(1 - p_t, gamma) * bce)

    return _loss
