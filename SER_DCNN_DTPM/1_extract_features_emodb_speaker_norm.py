import os
import librosa
import numpy as np

# Assuming EMOTION_MAP is defined elsewhere
from utility import EMOTION_MAP

np.random.seed(42)

curr_dir = os.path.dirname(os.path.abspath(__file__))

DATA_DIR    = os.path.join(curr_dir, "../emo_db/")
OUTPUT_DIR  = os.path.join(curr_dir, "./processed_emodb_speaker_norm/")

os.makedirs(OUTPUT_DIR, exist_ok=True)

train_speakers = ['03', '08', '09', '10', '11', '12', '13']
validation_speakers = ['14']
test_speakers = ['15', '16']

"""
We attempt to address: To develop deep learning techniques for SER invariant to speaker characteristics such as gender, age

This is an enhanced version of 1_extract_features_emodb.py that implements speaker-wise z-score normalization before slicing into segments.
The main difference is that we first calculate the mean and std for each speaker across all their audio, and then apply this normalization to each utterance before calculating deltas and slicing.
"""

# ==========================================
# STAGE 1: Extract Base Spectrograms & Group by Speaker
# ==========================================
print("Stage 1: Extracting raw log-Mels and calculating speaker statistics...")

speaker_mels = {} # To hold all audio data for each speaker
utterance_data = [] # To hold metadata and raw mel for later processing

for filename in os.listdir(DATA_DIR):
    if filename.endswith(".wav"):
        speaker_id = filename[:2]
        emotion_code = filename[5]
        
        if emotion_code not in EMOTION_MAP: 
            continue
            
        label = EMOTION_MAP[emotion_code]
        file_path = os.path.join(DATA_DIR, filename)
        
        # Load and extract just the base log-Mel
        y, sr = librosa.load(file_path, sr=16000)
        hop_length = int(sr * 0.010)
        win_length = int(sr * 0.025)
        
        mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=win_length, 
                                                  hop_length=hop_length, n_mels=64)
        log_mel = librosa.power_to_db(mel_spec, ref=np.max) # Shape: (64, Time)
        
        # Store for speaker stats
        if speaker_id not in speaker_mels:
            speaker_mels[speaker_id] = []
        speaker_mels[speaker_id].append(log_mel)
        
        # Store for Phase 2 processing
        utterance_data.append((filename, speaker_id, label, log_mel))

# Calculate Mean and Std for each speaker across all their audio
speaker_stats = {}
for spk, mels_list in speaker_mels.items():
    # Glue all this speaker's audio end-to-end to get the global frequency profile
    concatenated_mels = np.concatenate(mels_list, axis=1) 
    
    # Calculate mean and std for each of the 64 frequency bins (keepdims maintains the 2D shape)
    mean = np.mean(concatenated_mels, axis=1, keepdims=True)
    std = np.std(concatenated_mels, axis=1, keepdims=True) + 1e-8 # Add tiny number to prevent divide-by-zero
    
    speaker_stats[spk] = {'mean': mean, 'std': std}

print("Speaker statistics calculated successfully!")

# ==========================================
# STAGE 2: Normalize, Calculate Deltas, Slice, and Split
# ==========================================
print("Stage 2: Applying normalization and slicing segments...")

X_train, y_train, utterance_id_train = [], [], []
X_validation, y_validation, utterance_id_validation = [], [], []
X_test, y_test, utterance_id_test = [], [], []

for filename, speaker_id, label, raw_log_mel in utterance_data:
    
    # 1. Apply Speaker Z-Score Normalization
    spk_mean = speaker_stats[speaker_id]['mean']
    spk_std = speaker_stats[speaker_id]['std']
    normalized_mel = (raw_log_mel - spk_mean) / spk_std
    
    # 2. Calculate Deltas using the NORMALIZED spectrogram
    delta = librosa.feature.delta(normalized_mel)
    delta2 = librosa.feature.delta(normalized_mel, order=2)
    
    # Stack into a (3, 64, time_steps) array
    stacked_features = np.stack([normalized_mel, delta, delta2], axis=0)
    
    # 3. Slice into 64-frame segments with a 30-frame shift
    segment_frames = 64
    frame_shift = 30
    total_frames = stacked_features.shape[2]
    
    segments = []
    for start in range(0, total_frames - segment_frames + 1, frame_shift):
        end = start + segment_frames
        segment = stacked_features[:, :, start:end]
        segments.append(segment)
        
    # 4. Route to the correct dataset
    for seg in segments:
        if speaker_id in train_speakers:
            X_train.append(seg)
            y_train.append(label)
            utterance_id_train.append(filename)
        elif speaker_id in validation_speakers:
            X_validation.append(seg)
            y_validation.append(label)
            utterance_id_validation.append(filename)
        elif speaker_id in test_speakers:
            X_test.append(seg)
            y_test.append(label)
            utterance_id_test.append(filename)

# ==========================================
# STAGE 3: Save to Disk
# ==========================================
np.save(os.path.join(OUTPUT_DIR, "X_train.npy"), np.array(X_train))
np.save(os.path.join(OUTPUT_DIR, "y_train.npy"), np.array(y_train))
np.save(os.path.join(OUTPUT_DIR, "utterance_ids_train.npy"), np.array(utterance_id_train))
print(f"Extracted {len(X_train)} segments for training!")

np.save(os.path.join(OUTPUT_DIR, "X_validation.npy"), np.array(X_validation))
np.save(os.path.join(OUTPUT_DIR, "y_validation.npy"), np.array(y_validation))
np.save(os.path.join(OUTPUT_DIR, "utterance_ids_validation.npy"), np.array(utterance_id_validation))
print(f"Extracted {len(X_validation)} segments for validation!")

np.save(os.path.join(OUTPUT_DIR, "X_test.npy"), np.array(X_test))
np.save(os.path.join(OUTPUT_DIR, "y_test.npy"), np.array(y_test))
np.save(os.path.join(OUTPUT_DIR, "utterance_ids_test.npy"), np.array(utterance_id_test))
print(f"Extracted {len(X_test)} segments for testing!")