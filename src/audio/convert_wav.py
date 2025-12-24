
import os
import glob
import librosa
import soundfile as sf
import argparse
import traceback

# Try to import moviepy for mp4 handling
try:
    from moviepy import AudioFileClip
    HAS_MOVIEPY = True
except ImportError:
    HAS_MOVIEPY = False
    print("Warning: moviepy not found. .mp4 conversion may fail if system ffmpeg is missing.")

def convert_file(filepath):
    """Helper function to convert a single file."""
    try:
        print(f"Converting {filepath}...")
        
        ext = os.path.splitext(filepath)[1].lower()
        
        if ext == '.mp4':
            if HAS_MOVIEPY:
                # Use moviepy for mp4
                new_path = os.path.splitext(filepath)[0] + '.wav'
                with AudioFileClip(filepath) as clip:
                    clip.write_audiofile(new_path, logger=None)
                print(f"Saved {new_path}")
                return True
            else:
                # Fallback to librosa (will likely fail on Windows without ffmpeg)
                print("Attempting mp4 conversion with librosa (moviepy missing)...")
        
        # Default librosa load for audio files
        y, sr = librosa.load(filepath, sr=None)
        new_path = os.path.splitext(filepath)[0] + '.wav'
        sf.write(new_path, y, sr)
        print(f"Saved {new_path}")
        return True
        
    except Exception as e:
        # traceback.print_exc()
        print(f"Failed to convert {filepath}: {repr(e)}")
        return False

def convert_to_wav(path):
    """
    Scans the path for audio files (mp3, flac, m4a, mp4) and converts them to wav.
    Path can be a directory or a single file.
    """
    extensions = ['*.mp3', '*.flac', '*.m4a', '*.mp4']
    found_files = False

    # Case 1: Input is a single file
    if os.path.isfile(path):
        convert_file(path)
        return

    # Case 2: Input is a directory
    print(f"Scanning directory {path} for extensions: {extensions}...")
    
    for ext in extensions:
        # Recursive search in directory
        search_pattern = os.path.join(path, '**', ext)
        for filepath in glob.glob(search_pattern, recursive=True):
            found_files = True
            convert_file(filepath)
                
    if not found_files:
        print(f"No audio files found in {path} with extensions: {extensions}")
    else:
        print("Conversion completed.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert audio files to WAV.")
    parser.add_argument("--data_path", type=str, default="data/raw", help="Path to raw data directory or specific audio file")
    args = parser.parse_args()
    
    if os.path.exists(args.data_path):
        convert_to_wav(args.data_path)
    else:
        print(f"Path not found: {args.data_path}")
