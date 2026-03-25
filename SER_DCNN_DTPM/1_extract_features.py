import os
import librosa
import numpy as np

# EMO-DB Emotion mapping based on the 6th character of the filename
EMOTION_MAP = {'W': 0, 'L': 1, 'A': 2, 'F': 3, 'T': 4, 'E': 5, 'N': 6}
DATA_DIR = "./emodb/wav/" # Update this to your kaggle dataset path
OUTPUT_DIR = "./processed_data/"

os.makedirs(OUTPUT_DIR, exist_ok=True)

def process_audio(file_path):
    # The paper uses 16kHz sampling rate [cite: 371]
    y, sr = librosa.load(file_path, sr=16000)
    
    # Paper parameters: 25ms window, 10ms frame shift, 64 Mel-filter banks [cite: 194]
    hop_length = int(sr * 0.010) # 10ms
    win_length = int(sr * 0.025) # 25ms
    
    # 1. Static Log Mel-spectrogram (Channel 1)
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=win_length, 
                                              hop_length=hop_length, n_mels=64)
    log_mel = librosa.power_to_db(mel_spec, ref=np.max)
    
    # 2. Delta (Channel 2) and Delta-Delta (Channel 3) [cite: 203]
    delta = librosa.feature.delta(log_mel)
    delta2 = librosa.feature.delta(log_mel, order=2)
    
    # Stack into a (3, 64, time_steps) array
    stacked_features = np.stack([log_mel, delta, delta2], axis=0)
    
    # 3. Slice into 64-frame segments with a 30-frame shift [cite: 195]
    segment_frames = 64
    frame_shift = 30
    total_frames = stacked_features.shape[2]
    
    segments = []
    for start in range(0, total_frames - segment_frames + 1, frame_shift):
        end = start + segment_frames
        segment = stacked_features[:, :, start:end]
        segments.append(segment)
        
    return segments

# Process all files
X_data, y_labels, utterance_ids = [], [], []

for filename in os.listdir(DATA_DIR):
    if filename.endswith(".wav"):
        emotion_code = filename[5] # e.g., 'F' in 03a01Fa.wav
        if emotion_code not in EMOTION_MAP: continue
            
        label = EMOTION_MAP[emotion_code]
        segments = process_audio(os.path.join(DATA_DIR, filename))
        
        # We assign the same utterance label to every segment [cite: 402]
        for seg in segments:
            X_data.append(seg)
            y_labels.append(label)
            utterance_ids.append(filename) # Keep track of which segment belongs to which audio file

np.save(os.path.join(OUTPUT_DIR, "X_segments.npy"), np.array(X_data))
np.save(os.path.join(OUTPUT_DIR, "y_labels.npy"), np.array(y_labels))
np.save(os.path.join(OUTPUT_DIR, "utterance_ids.npy"), np.array(utterance_ids))
print(f"Extracted {len(X_data)} segments!")