import os
import json
import music21 as m21
from tensorflow import keras
import numpy as np
import subprocess
import tensorflow as tf
from sklearn.metrics import accuracy_score


KERN_DATASET_PATH = "essen/europa/deutschl/erk"
MISC_DATASET_PATH = "misc/"
SAVE_DIR = "dataset"

SINGLE_FILE_DATASET = "file_dataset"
SEQUENCE_LENGTH = 64
MAPPING_PATH = "mapping.json"

# durations are expressed in quarter length
ACCEPTABLE_DURATIONS = [
    0.25,  # 16th note
    0.5,  # 8th note
    0.75,
    1.0,  # quarter note
    1.5,
    2,  # half note
    3,
    4,  # whole note
]


def load_songs_in_kern(dataset_path):
    songs = []

    # go through all the files in dataset and load them with music21
    for path, subdirs, files in os.walk(dataset_path):
        for file in files:
            # consider only kern files
            if file[-3:] == "krn":
                song = m21.converter.parse(os.path.join(path, file))
                songs.append(song)
    return songs


def has_acceptable_durations(song, acceptable_durations):
    for note in song.flat.notesAndRests:
        if note.duration.quarterLength not in acceptable_durations:
            return False
    return True


def transpose(song):
    # get key from the song
    parts = song.getElementsByClass(m21.stream.Part)
    measures_part0 = parts[0].getElementsByClass(m21.stream.Measure)
    key = measures_part0[0][4]

    # estimate key using music21
    if not isinstance(key, m21.key.Key):
        key = song.analyze("key")

    # get interval for transposition. E.g., Bmaj -> Cmaj
    if key.mode == "major":
        interval = m21.interval.Interval(key.tonic, m21.pitch.Pitch("C"))
    elif key.mode == "minor":
        interval = m21.interval.Interval(key.tonic, m21.pitch.Pitch("A"))

    # transpose song by calculated interval
    tranposed_song = song.transpose(interval)
    return tranposed_song


def encode_song(song, time_stamp=0.25):
    # p = 60 , d = 1.0 ->[60, "_","_","_"]
    encoded_song = []
    for event in song.flat.notesAndRests:
        # handle notes
        if isinstance(event, m21.note.Note):
            symbol = event.pitch.midi  # 60
        # handle rests
        elif isinstance(event, m21.note.Rest):
            symbol = "r"
        # convert the note/rest into time series notation
        steps = int(event.duration.quarterLength / time_stamp)
        for step in range(steps):
            if step == 0:
                encoded_song.append(symbol)
            else:
                encoded_song.append("_")

    # cast encode song into string
    encoded_song = " ".join(map(str, encoded_song))

    return encoded_song


def preprocess(dataset_path):
    # load folk songs
    print("Loading songs...")
    songs = load_songs_in_kern(dataset_path)
    print(f"Loaded {len(songs)} songs.")

    for i, song in enumerate(songs):
        # filter out songs that have non-acceptable durations
        if not has_acceptable_durations(song, ACCEPTABLE_DURATIONS):
            continue

        # transpose songs to Cmaj/Amin
        song = transpose(song)

        # encode songs with music time series representation
        encoded_song = encode_song(song)

        # save songs to text file
        save_path = os.path.join(SAVE_DIR, str(i))
        with open(save_path, "w") as f:
            f.write(encoded_song)


def load(file_path):
    with open(file_path, "r") as f:
        song = f.read()
        return song


def create_single_file_dataset(dataset_path, file_dataset_path, sequence_length):
    new_song_delimiter = "/ " * sequence_length
    songs = ""

    # load encoded songs and add delimiters
    for path, _, files in os.walk(dataset_path):
        for file in files:
            file_path = os.path.join(path, file)
            song = load(file_path)
            songs = songs + song + " " + new_song_delimiter
    songs = songs[:-1]

    # save string that contains all datasets
    with open(file_dataset_path, "w") as f:
        f.write(songs)
    return songs


def create_mapping(songs, mapping_path):
    mappings = {}

    # identify the vocabulary
    songs = songs.split()
    vocabulary = list(set(songs))

    # create mappings
    for i, symbol in enumerate(vocabulary):
        mappings[symbol] = i

    # save vocabulary to a json file
    with open(mapping_path, "w") as f:
        json.dump(mappings, f, indent=4)


def convert_songs_to_int(songs):
    int_songs = []

    # load mappings
    with open(MAPPING_PATH, "r") as f:
        mappings = json.load(f)

    # cast songs string to a list
    songs = songs.split()

    # map songs to int
    for symbol in songs:
        int_songs.append(mappings[symbol])

    return int_songs


def generate_training_sequences(sequence_length):
    # [11, 12, 13, 14, ...] -> {i:[11,12] , t: 13 }; {i:[12,13] , t: 14}

    # load songs and map them to int
    songs = load(SINGLE_FILE_DATASET)
    int_songs = convert_songs_to_int(songs)

    # generate training sequences
    # Ex: 100 symbols, 64 sequence length, 100-64 =36
    inputs = []
    targets = []
    num_sequences = len(int_songs) - sequence_length
    for i in range(num_sequences):
        inputs.append(int_songs[i : i + sequence_length])
        targets.append(int_songs[i + sequence_length])

    # one-hot encode the sequences
    # inputs: (no. of sequences, sequence length, vocabulary size)
    # [ [0,1,2] [1,1,2] ] -> [ [1,0,0][0,1,0][0,0,1], [][][]]

    vocabulary_size = len(set(int_songs))
    inputs = keras.utils.to_categorical(inputs, num_classes=vocabulary_size)
    targets = np.array(targets)

    return inputs, targets


def open_midi_with_musescore(midi_file_path):
    try:
        # Replace 'MuseScore3' with 'MuseScore' if you're using an older version
        musescore_executable = r"C:\Program Files\MuseScore 4\bin\MuseScore4.exe"

        # Command to open the MIDI file in MuseScore
        command = [musescore_executable, midi_file_path]

        # Open MuseScore using subprocess
        subprocess.run(command, check=True)

        print(f"MIDI file '{midi_file_path}' opened successfully in MuseScore.")
    except subprocess.CalledProcessError as e:
        print(f"Error opening MIDI file: {e}")


def evaluate_saved_model(model_path, test_data, test_labels):
    # Load the saved model
    loaded_model = tf.keras.models.load_model(model_path)

    # Make predictions on the test data
    predictions = loaded_model.predict(test_data)

    # Assuming the model outputs probabilities, convert them to class predictions
    predicted_classes = tf.argmax(predictions, axis=1)

    # Calculate accuracy (or other metrics based on your problem)
    accuracy = accuracy_score(test_labels, predicted_classes)

    print(f"Accuracy: {accuracy}")


def main():
    preprocess(KERN_DATASET_PATH)
    songs = create_single_file_dataset(SAVE_DIR, SINGLE_FILE_DATASET, SEQUENCE_LENGTH)
    create_mapping(songs, MAPPING_PATH)
    # inputs, targets = generate_training_sequences(SEQUENCE_LENGTH)


if __name__ == "__main__":
    main()
