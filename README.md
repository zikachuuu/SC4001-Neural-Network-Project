# SC4001-Neural-Network-Project

## To Start
1. Create your conda env and configure it with `requirements.txt` and `environment.yml`

2. Download emo_db from https://www.kaggle.com/datasets/piyushagni5/berlin-database-of-emotional-speech-emodb

3. Place the wav files directly in `SC4001-Neural-Network-Project/emo_db` folder (no sub folders)

4. `cd` to SER_DCNN_DTPM and run the approriate preprocessing script.

Choice 1: Vanilla emo_db (no combination)

5. Run the bash files (`.sh`) in `run_1_extract_features_emodb`. You get the log mel segments ready to be fed into DCNN.

    - original: no LOSO CV, hardcoded first 7 speaker to train, 1 speaker to val, last 2 speaker to test. For quick run if running LOSO is taking too long.

    - loso: Leave one speaker out cross validation, since 10 speaker there are 10 folds (each fold 8 train, 1 val, 1 test). Should use this as final benchmarking.

    - normalized: normalized frequency and magnitude speaker-wise

    - augmented: created additional testcases by introducing some noises to frequency and magnitude

6. Run `2_dcnn_dtpm.ipynb`. Remenber to specify the dataset and splitmode at the top! The entire notebook can be run one shot, but note that the training of DCNN may take 1 to 2 hours. The trained DCNN model is saved so you can start from DTPM + SVM at another time. 

Choice 2: Combination of multiple audio files for dynamic SER

5. Run the bash files (`.sh`) in `run_1a_generate_dynamic_emodb_combination`. You get the combined audio files. Can stop here if you want to preprocess audio files differently from log mel segments.

    - default: may have repetition, each resulting audio file combined 2 to 4 audio files

    - unique: no repetition (chosen without replacement), each audio file combined 2 to 4 audio files

6. Run the bash files (`.sh`) in `run_1b_extract_features_emodb_comb`. You get the log mel segments ready to be fed into DCNN.

    - same sulfix, support for augmentation depreciated (its not effective)

