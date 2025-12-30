"""
Audio quality analysis utilities.
"""

import numpy as np


def calculate_snr(audio_array: np.ndarray, noise_floor_percentile=10) -> float:
    """
    Estimate Signal-to-Noise Ratio (dB).
    Uses percentile-based noise floor estimation.
    """
    if len(audio_array) == 0:
        return 0.0
        
    audio = audio_array.astype(np.float32)
    if np.max(np.abs(audio)) > 1.0:
        audio = audio / 32768.0
    
    # Estimate noise floor from quietest portions
    abs_audio = np.abs(audio)
    noise_floor = np.percentile(abs_audio, noise_floor_percentile)
    signal_level = np.percentile(abs_audio, 95)  # Peak signal
    
    if noise_floor < 1e-10:
        noise_floor = 1e-10
        
    snr = 20 * np.log10(signal_level / noise_floor)
    return max(0, min(60, snr))  # Clamp to reasonable range


def detect_clipping(audio_array: np.ndarray, threshold=0.99) -> float:
    """
    Detect percentage of samples that are clipped.
    Returns: 0.0 - 1.0 (percentage of clipped samples)
    """
    if len(audio_array) == 0:
        return 0.0
        
    audio = audio_array.astype(np.float32)
    if np.max(np.abs(audio)) > 1.0:
        audio = audio / 32768.0
        
    clipped = np.sum(np.abs(audio) >= threshold)
    return clipped / len(audio)


def calculate_rms(audio_array: np.ndarray) -> float:
    """Calculate RMS level in dB."""
    if len(audio_array) == 0:
        return -100.0
        
    audio = audio_array.astype(np.float32)
    if np.max(np.abs(audio)) > 1.0:
        audio = audio / 32768.0
        
    rms = np.sqrt(np.mean(audio ** 2))
    if rms < 1e-10:
        return -100.0
    return 20 * np.log10(rms)


def assess_quality(audio_array: np.ndarray) -> dict:
    """
    Comprehensive audio quality assessment.
    Returns dict with quality metrics and overall score.
    """
    snr = calculate_snr(audio_array)
    clipping = detect_clipping(audio_array)
    rms = calculate_rms(audio_array)
    
    # Quality scoring
    quality_score = 100
    issues = []
    
    # SNR assessment
    if snr < 10:
        quality_score -= 40
        issues.append("Çok düşük sinyal/gürültü oranı")
    elif snr < 20:
        quality_score -= 20
        issues.append("Düşük sinyal/gürültü oranı")
        
    # Clipping assessment
    if clipping > 0.05:
        quality_score -= 30
        issues.append("Ciddi ses kırpılması (clipping)")
    elif clipping > 0.01:
        quality_score -= 15
        issues.append("Hafif ses kırpılması")
        
    # Level assessment
    if rms < -40:
        quality_score -= 20
        issues.append("Ses seviyesi çok düşük")
    elif rms > -6:
        quality_score -= 10
        issues.append("Ses seviyesi çok yüksek")
        
    quality_score = max(0, min(100, quality_score))
    
    # Quality label
    if quality_score >= 80:
        label = "Mükemmel"
        color = "green"
    elif quality_score >= 60:
        label = "İyi"
        color = "blue"
    elif quality_score >= 40:
        label = "Kabul Edilebilir"
        color = "orange"
    else:
        label = "Düşük"
        color = "red"
    
    return {
        "snr_db": round(snr, 1),
        "clipping_percent": round(clipping * 100, 2),
        "rms_db": round(rms, 1),
        "score": quality_score,
        "label": label,
        "color": color,
        "issues": issues
    }
