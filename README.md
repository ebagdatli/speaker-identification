# Speaker Identification

A speaker identification system based on deep audio embeddings.

## Project Structure
- `data/`: Dataset storage (raw audio per person).
- `src/`: Source code.
  - `audio/`: Preprocessing and data loading.
  - `model/`: Neural network architecture.
  - `training/`: Training loop and utilities.
  - `inference/`: Speaker identification logic.
- `notebooks/`: Interactive tasks and experiments.
- `models/`: Saved model checkpoints.
- `embeddings/`: Database of speaker embeddings.

## Setup
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Add audio data to `data/raw/<person_name>/<audio_files>.wav`.

## Usage
See `notebooks/` for step-by-step guides.
