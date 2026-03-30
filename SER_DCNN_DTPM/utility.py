import numpy as np
import os

# EMO-DB Emotion mapping based on the 6th character of the filename
EMOTION_CODE_MAP = {
    'W': 0,     # 'W' for "Wut" (Anger)
    'L': 1,     # 'L' for "Langeweile" (Boredom)
    'A': 2,     # 'A' for "Angst" (Fear)
    'F': 3,     # 'F' for "Freude" (Happiness)
    'T': 4,     # 'T' for "Trauer" (Sadness)
    'E': 5,     # 'E' for "Ekel" (Disgust)
    'N': 6      # 'N' for "Neutrale Emotion" (Neutral)
}

EMOTION_ENG_MAP = {
    0: 'Anger',
    1: 'Boredom',
    2: 'Fear',
    3: 'Happiness',
    4: 'Sadness',
    5: 'Disgust',
    6: 'Neutral'
}

# 1. Tweaking Amplitude (The Z-Axis / "Brightness")
# In a log-Mel spectrogram, the values represent decibels (volume). Because it is on a logarithmic scale, 
# adding a constant is equivalent to multiplying the raw audio volume, and adding Gaussian noise simulates background static.

# A. Volume Scaling (Shift)
def augment_volume(log_mel, max_db_shift=5.0):
    # Randomly make the entire clip louder or quieter by up to 5dB
    shift = np.random.uniform(-max_db_shift, max_db_shift)
    return log_mel + shift

# B. Gaussian Noise (Static)
def add_matrix_noise(log_mel, noise_level=0.5):
    # Generate random static matching the matrix shape
    noise = np.random.normal(0, noise_level, log_mel.shape)
    return log_mel + noise


# 2. Tweaking Frequency (The Y-Axis)
# Your Y-axis has 64 rows representing frequency bins (from low bass at row 0 to high treble at row 63).

# A. Pitch Shifting (Matrix Translation)
# To make a voice sound deeper or higher, you simply shift all the rows up or down. 
# Crucial detail: Do not use np.roll, or the high frequencies will wrap around to the bottom! You must pad the empty space with the matrix's minimum value (silence).
def matrix_pitch_shift(log_mel, shift_bins):
    # shift_bins > 0 shifts pitch up; < 0 shifts pitch down
    shifted = np.full_like(log_mel, log_mel.min())
    if shift_bins > 0:
        shifted[shift_bins:, :] = log_mel[:-shift_bins, :]
    elif shift_bins < 0:
        shift_bins = abs(shift_bins)
        shifted[:-shift_bins, :] = log_mel[shift_bins:, :]
    else:
        shifted = log_mel.copy()
    return shifted

# B. Frequency Masking (SpecAugment)
# This forces the neural network to recognize an emotion even if a specific frequency band is missing (making it highly robust to different microphones or voices).
def freq_mask(log_mel, max_mask_width=8):
    # Randomly zero out a horizontal band of frequencies
    masked = log_mel.copy()
    f, t = masked.shape
    
    mask_width = np.random.randint(1, max_mask_width)
    f0 = np.random.randint(0, f - mask_width)
    
    # Set the horizontal block to "silence" (the minimum value)
    masked[f0:f0 + mask_width, :] = masked.min()
    return masked

from scipy.ndimage import zoom

# 3. Tweaking Time (The X-Axis)
# Your X-axis represents chronological time.

# A. Time Stretching (Matrix Interpolation)
# To make someone speak 20% faster or slower, you literally stretch or squash the matrix horizontally. 
# Because it's a 2D array, we can use SciPy's zoom function to interpolate the new pixels.
def matrix_time_stretch(log_mel, rate):
    # rate > 1.0 stretches time (slower speech); rate < 1.0 squishes time (faster speech)
    # We only zoom along the time axis (axis 1), keeping frequency (axis 0) at 1.0
    return zoom(log_mel, (1.0, rate))

# B. Time Masking (SpecAugment)
# This zeroes out a vertical block of time, simulating a dropped audio signal or a sudden burst of overriding noise. 
# It forces the DTPM algorithm to rely on the rest of the timeline.
def time_mask(log_mel, max_mask_width=15):
    # Randomly zero out a vertical band of time
    masked = log_mel.copy()
    f, t = masked.shape
    
    mask_width = np.random.randint(1, max_mask_width)
    t0 = np.random.randint(0, t - mask_width)
    
    # Set the vertical block to "silence"
    masked[:, t0:t0 + mask_width] = masked.min()
    return masked