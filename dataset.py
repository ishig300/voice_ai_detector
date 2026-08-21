import os
from datasets import Dataset


def build_dataset():
    # list that will hold all training samples as dictionaries
    # each sample = {audio_path, label}
    data = []

    # os.listdir returns all filenames in a directory
    # here, it gets all human audio files (e.g., "h1.wav", "h2.wav")
    for file in os.listdir("data/human"):
        data.append({
            # full relative path so later preprocessing can load the audio file
            "audio_path": "data/human/" + file,

            # label 0 represents HUMAN speech
            # model will learn to associate these audio patterns with class 0
            "label": 0
        })

    # repeat the same process for AI-generated speech samples
    for file in os.listdir("data/ai"):
        data.append({
            # path to AI-generated audio file
            "audio_path": "data/ai/" + file,

            # label 1 represents AI-generated speech
            # this includes all TTS systems (ElevenLabs, Polly, etc.)
            "label": 1
        })

    # convert Python list into Hugging Face Dataset object
    # this enables compatibility with transformers training pipelines
    # and allows operations like batching, mapping, and train/test splitting
    ds = Dataset.from_list(data)

    # split into training (80%) and testing (20%) sets
    # seed=42 makes the split reproducible, so we get the same
    # train/test groups every time we run this file
    ds = ds.train_test_split(test_size=0.2, seed=42)

    # ds is now a DatasetDict with two keys: "train" and "test"
    # every other file that calls build_dataset() gets the split for free
    return ds


if __name__ == "__main__":
    # runs only when this file is executed directly (not imported)
    # useful for testing that dataset creation works correctly

    # build dataset from audio folders (already split into train/test)
    dataset = build_dataset()

    # print dataset summary (number of rows, features, structure)
    print(dataset)
