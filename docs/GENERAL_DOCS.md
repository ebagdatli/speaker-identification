# Speaker Identification: Project Overview

Welcome to the **Speaker Identification** project documentation. This guide provides a high-level overview of what the project does and how to use it.

## What is this project?
This is a machine learning system designed to **recognize people by their voice**. It is not a "speech-to-text" system; rather, it identifies *who* is speaking, conceptually similar to FaceID but for audio.

It is capable of:
1.  **Enrolling** new users (learning their voice from a few audio samples).
2.  **Identifying** users from a new audio recording.
3.  **Rejecting** unknown speakers.

## How does it work? (Simplified)
The system works by converting audio into a unique "fingerprint" (embedding vector).
1.  **Input**: You provide an audio file (Format: WAV, MP3, MP4, etc.).
2.  **Processing**: The system converts this audio into a visual representation (Spectrogram).
3.  **Fingerprinting**: A neural network scans this visual and extracts a numerical code (Embedding).
    *   *For details on the Neural Network, see [Technical Docs: Model Architecture](TECHNICAL_DOCS.md#2-model-architecture-speakerencoder)*.
4.  **Matching**: This code is compared against a database of known codes. If it matches closely enough (> 85%), the name is returned.

## Project Organization

The project is divided into 4 main stages, which correspond to the notebooks provided:

1.  **Data Preparation** (`01_data_preparation.ipynb`)
    *   Organizing your raw audio files into folders per person.
    *   *See [Technical Docs: Preprocessing](TECHNICAL_DOCS.md#4-preprocessing--data-loading)*.
2.  **Training** (`02_training.ipynb`)
    *   Teaching the model to distinguish between the voices you provided.
    *   Uses advanced "Contrastive Learning" to learn robustly.
    *   *See [Technical Docs: Loss Function](TECHNICAL_DOCS.md#3-loss-function-ntxent_loss)*.
3.  **Enrollment** (`03_embedding_generation.ipynb`)
    *   Creating "ID cards" (embeddings) for your known speakers using the trained model.
4.  **Inference / Testing** (`04_inference.ipynb` or UI)
    *   Real-world testing.
    *   *See [Technical Docs: Inference Logic](TECHNICAL_DOCS.md#5-inference-logic)*.

## How to use the UI
We have provided a modern web interface for easy testing.
1.  Run the command: `streamlit run streamlit_app.py`
2.  Open your browser to the provided URL.
3.  Click the **Mic** icon to record yourself.
4.  The system will instantly tell you who you are!

## Next Steps for Developers
If you want to understand the code structure, modify the neural network, or change the training logic, please proceed to the **[Detailed Technical Documentation](TECHNICAL_DOCS.md)**.
