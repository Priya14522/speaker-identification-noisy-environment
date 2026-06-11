import os
import numpy as np
import librosa
import random
import joblib
import matplotlib.pyplot as plt

from hmmlearn import hmm
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# =====================================
# CONFIGURATION
# =====================================

DATASET_PATH = r"D:\speaker_recognition\train"

N_MFCC = 13
N_COMPONENTS = 16
N_ITER = 100
TRAIN_FILES = 300
RANDOM_STATE = 42

# =====================================
# FEATURE EXTRACTION
# =====================================

def extract_features(file_path):

    audio, sr = librosa.load(file_path, sr=None)

    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=N_MFCC)

    # CMVN
    mfcc = (mfcc - np.mean(mfcc, axis=1, keepdims=True)) / \
           (np.std(mfcc, axis=1, keepdims=True) + 1e-8)

    return mfcc.T


# =====================================
# NOISE ADDITION
# =====================================

def add_white_noise(signal, snr_db):

    signal_power = np.mean(signal ** 2)

    snr_linear = 10 ** (snr_db / 10)

    noise_power = signal_power / snr_linear

    noise = np.random.normal(0, np.sqrt(noise_power), signal.shape)

    return signal + noise


# =====================================
# SPECTRAL SUBTRACTION
# =====================================

def spectral_subtraction_denoise(signal, sr):

    stft = librosa.stft(signal, n_fft=1024, hop_length=512)

    magnitude, phase = np.abs(stft), np.angle(stft)

    noise_frames = max(1, int(0.5 * sr / 512))

    noise_spectrum = np.mean(magnitude[:, :noise_frames], axis=1, keepdims=True)

    subtracted = magnitude - noise_spectrum

    subtracted = np.maximum(subtracted, 0)

    reconstructed = subtracted * np.exp(1j * phase)

    denoised = librosa.istft(reconstructed, hop_length=512)

    return denoised


# =====================================
# NOISY + DENOISED FEATURES
# =====================================

def extract_noisy_denoised_features(file_path, snr_db):

    audio, sr = librosa.load(file_path, sr=None)

    noisy_audio = add_white_noise(audio, snr_db)

    denoised_audio = spectral_subtraction_denoise(noisy_audio, sr)

    mfcc = librosa.feature.mfcc(y=denoised_audio, sr=sr, n_mfcc=N_MFCC)

    # CMVN
    mfcc = (mfcc - np.mean(mfcc, axis=1, keepdims=True)) / \
           (np.std(mfcc, axis=1, keepdims=True) + 1e-8)

    return mfcc.T


# =====================================
# LOAD DATASET
# =====================================

def load_data(dataset_path):

    X_train = {}
    X_test = {}

    random.seed(RANDOM_STATE)

    for speaker in os.listdir(dataset_path):

        speaker_path = os.path.join(dataset_path, speaker)

        if not os.path.isdir(speaker_path):
            continue

        files = [
            os.path.join(speaker_path, f)
            for f in os.listdir(speaker_path)
            if f.endswith(".wav")
        ]

        random.shuffle(files)

        split_index = int(0.8 * len(files))

        train_files = files[:split_index]
        test_files = files[split_index:]

        X_train[speaker] = train_files
        X_test[speaker] = test_files

        print(f"{speaker} -> Train: {len(train_files)} | Test: {len(test_files)}")

    return X_train, X_test

# =====================================
# TRAIN MODELS
# =====================================

def train_models(X_train):

    models = {}

    y_true = []
    y_pred = []

    for speaker in X_train:

        print(f"\nTraining model for {speaker}")

        features = []
        
        lengths = []

        for file in X_train[speaker]:

            mfcc = extract_features(file)

            features.append(mfcc)

            lengths.append(len(mfcc))

        features = np.vstack(features)

        model = hmm.GaussianHMM(
            n_components=N_COMPONENTS,
            covariance_type='diag',
            n_iter=N_ITER,
            random_state=RANDOM_STATE
        )

        model.fit(features, lengths)

        models[speaker] = model

    joblib.dump(models, "speaker_models.pkl")

    print("\nModels Saved!")

    # -------- TRAINING ACCURACY --------

    for speaker in X_train:

        for file in X_train[speaker]:

            mfcc = extract_features(file)

            scores = {}

            for model_speaker in models:

                scores[model_speaker] = models[model_speaker].score(mfcc)

            pred = max(scores, key=scores.get)

            y_true.append(speaker)
            y_pred.append(pred)

    train_acc = accuracy_score(y_true, y_pred)

    print("\nTraining Accuracy:", train_acc * 100)

    return models


# =====================================
# TESTING
# =====================================

def test_models(models, X_test, snr_db):

    y_true = []

    y_pred = []

    print(f"\nTesting with Noise + Spectral Subtraction (SNR={snr_db} dB)")

    for speaker in X_test:

        for file in X_test[speaker]:

            mfcc = extract_noisy_denoised_features(file, snr_db)

            scores = {}

            for model_speaker in models:

                scores[model_speaker] = models[model_speaker].score(mfcc)

            pred = max(scores, key=scores.get)

            y_true.append(speaker)

            y_pred.append(pred)

    acc = accuracy_score(y_true, y_pred)

    print("Testing Accuracy:", acc * 100)

    cm = confusion_matrix(y_true, y_pred)

    print("\nConfusion Matrix\n")

    print(cm)

    print("\nPrecision Recall F1 Score\n")

    print(classification_report(y_true, y_pred))

    return acc


# =====================================
# ACCURACY vs SNR GRAPH
# =====================================

def plot_accuracy_vs_snr(snr_levels, accuracies):

    plt.figure()

    plt.plot(snr_levels, accuracies, marker='o')

    plt.xlabel("SNR (dB)")

    plt.ylabel("Accuracy")

    plt.title("Accuracy vs Noise Level")

    plt.grid(True)

    plt.show()


# =====================================
# MAIN
# =====================================

def main():

    X_train, X_test = load_data(DATASET_PATH)

    models = train_models(X_train)

    snr_levels = [60, 50, 40, 30, 20]

    accuracies = []

    print("\n========== ROBUSTNESS TEST ==========")

    for snr in snr_levels:

        acc = test_models(models, X_test, snr)

        accuracies.append(acc)

    print("\nSNR vs Accuracy")

    for s, a in zip(snr_levels, accuracies):

        print(f"SNR {s} dB -> {a*100:.2f}%")

    plot_accuracy_vs_snr(snr_levels, accuracies)


if __name__ == "__main__":
    main()