import numpy as np
# library for data science
import librosa
# library for audio analysis

# clips should fall somewhere between these two lengths, in seconds
MIN_DURATION = 2
MAX_DURATION = 5

# sample rate we want every clip to use
# audio files can be recorded at different sample rates (e.g. 44100, 48000)
# so we resample everything to the same rate to keep features consistent
SAMPLE_RATE = 16000

# how many MFCC coefficients to extract per audio file
# MFCCs (Mel-Frequency Cepstral Coefficients) describe the "shape" of a sound
# they're a common way to turn raw audio into numbers a model can learn from
N_MFCC = 40


def load_audio(audio_path):
    # librosa.load reads a WAV file and returns:
    #   signal: a NumPy array of audio samples
    #   sr: the sample rate the audio was loaded at
    # sr=SAMPLE_RATE forces librosa to resample the audio to our target rate
    # mono=True mixes stereo audio down to a single channel
    signal, sr = librosa.load(audio_path, sr=SAMPLE_RATE, mono=True)

    # convert our min/max seconds into a number of samples, since that's
    # what the audio array is actually measured in
    min_samples = int(MIN_DURATION * SAMPLE_RATE)
    max_samples = int(MAX_DURATION * SAMPLE_RATE)

    if len(signal) > max_samples:
        # if the clip is longer than MAX_DURATION seconds, cut it down
        signal = signal[:max_samples]
    elif len(signal) < min_samples:
        # if the clip is shorter than MIN_DURATION seconds, pad the end
        # with zeros (silence) so it reaches the minimum length
        # np.pad adds extra values without changing the real audio content
        padding = min_samples - len(signal)
        signal = np.pad(signal, (0, padding))
    # if the clip is already between MIN_DURATION and MAX_DURATION seconds,
    # leave it alone - extract_features() below always outputs a fixed-size
    # vector no matter how many samples go in, so this variability is fine

    return signal


def extract_features(audio_path):
    # load and standardize the raw audio samples
    signal = load_audio(audio_path)

    # librosa.feature.mfcc turns the raw waveform into N_MFCC coefficients
    # per audio frame, so the raw output shape is (N_MFCC, number_of_frames)
    mfccs = librosa.feature.mfcc(y=signal, sr=SAMPLE_RATE, n_mfcc=N_MFCC)

    # different clips can produce a different number of frames
    # taking the mean across the time axis collapses this into
    # a single fixed-length vector of size N_MFCC, no matter how long the clip was
    features = np.mean(mfccs, axis=1)

    return features


def preprocess_example(example):
    # this function is designed to be used with dataset.map()
    # "example" is one row of the Hugging Face Dataset, e.g.:
    # {"audio_path": "data/human/hello.wav", "label": 0}

    # extract_features loads the file from disk and converts it into MFCC features
    # this is the "lazy loading" step - audio is only read now, not earlier
    example["features"] = extract_features(example["audio_path"])

    return example


def preprocess_dataset(dataset):
    # dataset.map() runs preprocess_example() on every row
    # and returns a new Dataset with an extra "features" column added
    return dataset.map(preprocess_example)


if __name__ == "__main__":
    # runs only when this file is executed directly (not imported)
    # useful for testing that feature extraction works correctly

    from dataset import build_dataset

    # build_dataset() now returns a DatasetDict already split into
    # "train" and "test", so we grab the train split to test on
    ds = build_dataset()

    # run preprocessing to add MFCC features to every example
    train_ds = preprocess_dataset(ds["train"])

    # print the first row so we can see what a processed example looks like
    print(train_ds[0])
