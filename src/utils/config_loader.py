"""
Central configuration loader for the Speaker Identification system.
"""

import os
import yaml
from dataclasses import dataclass, field
from typing import Optional

# Find config file relative to this file or project root
_CONFIG_FILE = None
_CONFIG_CACHE = None


def _find_config_file():
    """Find config.yaml in project root."""
    global _CONFIG_FILE
    if _CONFIG_FILE:
        return _CONFIG_FILE
        
    # Try relative to this file
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Go up to project root (src/utils -> src -> root)
    for _ in range(3):
        config_path = os.path.join(current_dir, "config.yaml")
        if os.path.exists(config_path):
            _CONFIG_FILE = config_path
            return config_path
        current_dir = os.path.dirname(current_dir)
    
    # Try current working directory
    cwd_config = os.path.join(os.getcwd(), "config.yaml")
    if os.path.exists(cwd_config):
        _CONFIG_FILE = cwd_config
        return cwd_config
        
    return None


def load_config() -> dict:
    """Load configuration from config.yaml with caching."""
    global _CONFIG_CACHE
    if _CONFIG_CACHE is not None:
        return _CONFIG_CACHE
        
    config_path = _find_config_file()
    
    if config_path and os.path.exists(config_path):
        with open(config_path, 'r', encoding='utf-8') as f:
            _CONFIG_CACHE = yaml.safe_load(f)
            return _CONFIG_CACHE
    
    # Return defaults if no config file
    _CONFIG_CACHE = get_default_config()
    return _CONFIG_CACHE


def get_default_config() -> dict:
    """Return default configuration."""
    return {
        'audio': {
            'sample_rate': 8000,
            'duration': 1.0,
            'hop_length': 256,
            'n_fft': 1024,
            'n_mels': 256
        },
        'training': {
            'batch_size': 32,
            'epochs': 30,
            'lr': 0.0005,
            'val_split': 0.1
        },
        'inference': {
            'threshold': 0.82,
            'rolling_window_size': 2.0,
            'vad_aggressiveness': 0.5,
            'similarity_metric': 'cosine'
        },
        'paths': {
            'model_path': 'models/speaker_encoder.pt',
            'embeddings_path': 'embeddings/speakers.json',
            'noise_path': 'data/noise',
            'ir_path': 'data/ir'
        }
    }


def get(key: str, default=None):
    """
    Get a config value using dot notation.
    Example: get('audio.sample_rate') -> 8000
    """
    config = load_config()
    keys = key.split('.')
    value = config
    
    for k in keys:
        if isinstance(value, dict) and k in value:
            value = value[k]
        else:
            return default
    
    return value


# Convenience accessors
def audio_config() -> dict:
    return load_config().get('audio', {})

def training_config() -> dict:
    return load_config().get('training', {})

def inference_config() -> dict:
    return load_config().get('inference', {})

def paths_config() -> dict:
    return load_config().get('paths', {})


# Pre-defined getters for common values
def sample_rate() -> int:
    return get('audio.sample_rate', 8000)

def duration() -> float:
    return get('audio.duration', 1.0)

def model_path() -> str:
    return get('paths.model_path', 'models/speaker_encoder.pt')

def embeddings_path() -> str:
    return get('paths.embeddings_path', 'embeddings/speakers.json')

def threshold() -> float:
    return get('inference.threshold', 0.82)

def vad_confidence() -> float:
    return get('inference.vad_aggressiveness', 0.5)
