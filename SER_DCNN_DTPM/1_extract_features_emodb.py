import os
import librosa
import numpy as np

from utility import EMOTION_MAP

DATA_DIR = "../emo_db/" # Update this to your kaggle dataset path
OUTPUT_DIR = "./processed_data_emodb/"

os.makedirs(OUTPUT_DIR, exist_ok=True)

def process_audio(file_path):
    # The paper uses 16kHz sampling rate
    y, sr = librosa.load(file_path, sr=16000)
    
    # Paper parameters: 25ms window, 10ms frame shift, 64 Mel-filter banks
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


# We have 10 speakers in EMO-DB
# 03 - male, 31 years old
# 08 - female, 34 years
# 09 - female, 21 years
# 10 - male, 32 years
# 11 - male, 26 years
# 12 - male, 30 years
# 13 - female, 32 years
# 14 - female, 35 years
# 15 - male, 25 years
# 16 - female, 31 years

# We will use speakers 03, 08, 09, 10, 11, 12, 13 for training (7 speakers)
# Speaker 14 for validation (1 speaker)
# Speakers 15, 16 for testing (2 speakers)

# If the same speaker appeared in both sets, then it may lead to data leakage and overfitting, 
# as the model could learn speaker-specific features rather than emotion-specific features

X_train = []
y_train = []
utterance_id_train = [] # To keep track of which segment belongs to which audio file

X_validation = []
y_validation = []
utterance_id_validation = [] # To keep track of which segment belongs to which audio file

X_test = []
y_test = []
utterance_id_test = [] # To keep track of which segment belongs to which audio file


train_speakers = ['03', '08', '09', '10', '11', '12', '13']
validation_speakers = ['14']
test_speakers = ['15', '16']

for filename in os.listdir(DATA_DIR):
    if filename.endswith(".wav"):
        speaker_id      = filename[:2]  # e.g., '03' in 03a01Fa.wav
        emotion_code    = filename[5]   # e.g., 'F' in 03a01Fa.wav
        if emotion_code not in EMOTION_MAP: 
            raise ValueError(f"Unknown emotion code '{emotion_code}' in filename '{filename}'")
            
        label = EMOTION_MAP[emotion_code]
        segments = process_audio(os.path.join(DATA_DIR, filename))
        
        # We assign the same utterance label to every segment [cite: 402]
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