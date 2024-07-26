import keras
import json
import numpy as np
import music21 as m21
from pre_process import SEQUENCE_LENGTH, MAPPING_PATH, open_midi_with_musescore


class MelodyGenerator:
    def __init__(self, model_path="model.h5"):
        self.model_path = model_path
        self.model = keras.models.load_model(model_path)

        with open(MAPPING_PATH, "r") as f:
            self._mappings = json.load(f)

        self._start_symbols = ["/"] * SEQUENCE_LENGTH

    def generate_melody(self, seed, num_steps, max_sequence_length, temperature):
        # seed :"64_63_ _ _"

        # create seed with start symbols
        seed = seed.split()
        melody = seed
        seed = self._start_symbols + seed

        # map seed to int
        seed = [self._mappings[symbol] for symbol in seed]

        for _ in range(num_steps):
            # limit the seed to max_sequence_length
            seed = seed[-max_sequence_length:]

            # one-hot encode the seed
            onehot_seed = keras.utils.to_categorical(
                seed, num_classes=len(self._mappings)
            )
            # size of this = (max_sequence_length, num of symbols in the vocabulary)
            # we have to convert it to size = (1,max_sequence_length, num of symbols in the vocabulary)  [because of the keras(it wants in 3 dim)]
            onehot_seed = onehot_seed[np.newaxis, ...]

            # make a prediction
            probabilitites = self.model.predict(onehot_seed)[0]
            # [0.1,0.2,0.1,0.6] -> 1
            output_int = self._sample_with_temperature(probabilitites, temperature)

            # update a seed
            seed.append(output_int)

            # map int to the midi value
            output_symbol = [k for k, v in self._mappings.items() if v == output_int][0]

            # check whether we're a the end of a melody
            if output_symbol == "/":
                break

            # update a melody
            melody.append(output_symbol)
        return melody

    def _sample_with_temperature(self, probabilitites, temperature):
        # temperature -> infinity
        # temperature -> 0
        # temperature -> 1
        predictions = np.log(probabilitites) / temperature
        probabilitites = np.exp(predictions) / np.sum(np.exp(predictions))

        choices = range(len(probabilitites))  # [0,1,2,3]
        index = np.random.choice(choices, p=probabilitites)

        return index

    def save_melody(
        self, melody, step_duration=0.25, format="midi", file_name="mel.mid"
    ):
        # create a music21 stream
        stream = m21.stream.Stream()

        start_symbol = None
        step_counter = 1

        # parse all the symbols in the melody and create note/rest objects
        for i, symbol in enumerate(melody):
            # handle case in which we have a note/rest
            if symbol != "_" or i + 1 == len(melody):
                # ensure we're dealing with note/rest beyond the first one
                if start_symbol is not None:
                    quarter_length_duration = (
                        step_duration * step_counter
                    )  # 0.25 * 4 = 1

                    # handle rest
                    if start_symbol == "r":
                        m21_event = m21.note.Rest(quarterLength=quarter_length_duration)

                    # handle note
                    else:
                        m21_event = m21.note.Note(
                            int(start_symbol), quarterLength=quarter_length_duration
                        )

                    stream.append(m21_event)

                    # reset the step counter
                    step_counter = 1

                start_symbol = symbol

            # handle case in which we have a prolongation sign "_"
            else:
                step_counter += 1

        # write the m21 stream to a midi file
        stream.write(format, file_name)


if __name__ == "__main__":
    mg = MelodyGenerator()
    seed = "55 _ 60 _ 60 _ 62 _ 62 _ 64 _ "
    melody = mg.generate_melody(seed, 500, SEQUENCE_LENGTH, 0.8)
    print(melody)
    mg.save_melody(melody=melody)
    open_midi_with_musescore("mel.mid")
