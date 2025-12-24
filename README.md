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

### 1. Data Preparation
The system relies on audio data in `data/raw`. Structure your data as follows:

```
data/
  raw/
    Ahmet/
      audio1.wav
      audio2.wav
    Mehmet/
      rec1.wav
```

If you have files in other formats (`.mp3`, `.flac`, `.m4a`, `.mp4`), you can automatically convert them using the provided script:

```bash
python src/audio/convert_wav.py
```
*Note: This script will verify conversion and delete the original source files to verify output.*

### 2. Training
To train the model:
1. Open `notebooks/02_training.ipynb`.
2. Run all cells.
   - Reads data from `data/raw`.
   - Trains the model.
   - Saves checkpoint to `models/speaker_encoder.pt`.

### 3. Embedding Generation (Enrollment)
Create embeddings for the known speakers:
1. Open `notebooks/03_embedding_generation.ipynb`.
2. Run all cells.
   - Generates and saves embeddings to `embeddings/speakers.json`.

### 4. Inference
Identify a speaker from a new audio file:
1. Open `notebooks/04_inference.ipynb`.
2. Set `test_file` to your audio path.
3. Run the cells to see the result.

