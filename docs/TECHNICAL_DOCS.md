# Technical Documentation: Speaker Identification

This document provides a deep-dive into the technical implementation, architectural choices, and specific classes used in the project.

## 1. System Overview

The project implements a **Metric Learning** approach for speaker identification. Instead of classifying audio directly into a fixed set of classes (which would require retraining for every new user), the system learns to map audio segments to a high-dimensional vector space (embedding space).

-   **Objective**: Ensure embeddings from the same speaker are close together (high cosine similarity), while embeddings from different speakers are far apart.
-   **Method**: Contrastive Learning using NTxent Loss (SimCLR framework).

## 2. Model Architecture: `SpeakerEncoder`

**File**: `src/model/speaker_encoder.py`

We utilize a Convolutional Neural Network (CNN) adapted from the "Deep Audio Fingerprinting" paper (Neural Fingerprinter).

### Why this architecture?
1.  **Separable Convolutions**: The model uses Depthwise Separable Convolutions (`SC` class). This significantly reduces the number of parameters and computation cost compared to standard convolutions, making the model CPU-friendly suitable for real-time inference.
2.  **Spectro-Temporal Features**: The kernel sizes and strides are designed to capture both spectral (frequency) and temporal (time) patterns in the Mel Spectrogram.
3.  **Global Pooling**: The reference architecture used a "Divide and Encode" layer for hashing. We modified this to use a flattened linear projection to generate a continuous, fixed-size dense embedding (256-d), which is ideal for similarity search.

### Class: `SpeakerEncoder`
-   **Input**: Mel Spectrogram `(Batch, 1, 256, 32)`
    -   128 time steps ~ 1 second of audio.
    -   256 Mel frequency bins.
-   **Backbone**: 7 layers of Separable Convolutions.
-   **Head**: Flatten -> Linear(2048, 256) -> L2 Normalization.
-   **Output**: 256-dimensional unit vector.

## 3. Loss Function: `NTxent_Loss`

**File**: `src/training/loss.py`

We switched from Triplet Loss to **Normalized Temperature-scaled Cross Entropy Loss (NTxent)**.

### Why not Triplet Loss?
Triplet loss requires careful mining of "hard" negatives (triplets where the model makes mistakes) to be effective. Without mining, training can be unstable or slow.

### Why NTxent (SimCLR)?
NTxent maximizes the similarity between two augmented views of the *same* sample (positive pair) while minimizing similarity with *all other* samples in the batch (negatives).
-   **Effectiveness**: It learns very robust representations by seeing many negatives at once (batch size dependent).
-   **Robustness**: By augmenting the views (noise, time shift), the model learns to ignore channel noise and focus on voice characteristics.

## 4. Preprocessing & Data Loading

**File**: `src/audio/preprocessing.py`

### Class: `SpeakerDataset`
This dataset is designed for multiple-view training.
-   **Anchor Selection**: Picks a random audio file for a speaker.
-   **View Generation**:
    1.  **View 1**: A random 1-second chunk from File A.
    2.  **View 2**: A random 1-second chunk from File B (same speaker) OR a different chunk from File A.
-   **Augmentation**: View 2 is heavily augmented using the `audio_augmentation_chain`.

### Augmentation Chain
To prevent overfitting, we apply:
1.  **Background Noise**: Mixing with environmental sounds (SNR 0-15dB).
2.  **Impulse Response**: Simulating room reverb.
3.  **Time Modulations**: Shifting start times.
4.  **Frequency Masking**: Simulating bad microphones.

## 5. Inference Logic

**File**: `src/inference/identify_speaker.py`

### Class: `SpeakerIdentifier`
Handles the full lifecycle of using the model.

-   **`enroll_speaker(name, path)`**:
    1.  Computes embedding for the audio.
    2.  Stores it in a JSON database (`embeddings/speakers.json`).
-   **`identify(path)`**:
    1.  **Sliding Window**: It does *not* just check one segment. It slides a 1-second window over the entire input file (with overlap).
    2.  **Averaging**: The embeddings for all windows are averaged. This creates a highly stable "fingerprint" of the whole utterance, reducing the chance of a random noise spike causing a misidentification.
    3.  **Thresholding**: Compares the averaged embedding against all enrolled speakers using Cosine Similarity. If the best score < `0.85` (tunable), it returns "Unknown".

## 6. Directory Structure Summary

| Directory | Purpose |
|Data | Raw and Processed audio files.|
|Embeddings | JSON storage for enrolled speaker vectors.|
|Models | Checkpoints (`.pt` files).|
|Notebooks | Jupyter notebooks for step-by-step execution.|
|Src | Source code modules.|
