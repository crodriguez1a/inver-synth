import math
from functools import lru_cache

import numpy as np
import samplerate
from keras import Model

# import keras
from scipy.io import wavfile
from tensorflow import keras


class SoundfileGenerator(keras.utils.Sequence):
    """
    Generator that pulls data out of an audio file. It will:
    - resample the data to the right format
    - give it out in chunks of the right size
    - optionally turn each chunk of audio into output labels with a given model
    TODO: work with multiple files
    """

    def __init__(
        self,
        audio_file: str,
        model: Model = None,
        batch_size=32,
        n_samps=16384,
        sample_rate=16384,
        shuffle=True,
        last: float = 0.0,
        first: float = 0.0,
        channels_last=False,
    ):
        "Initialization"
        self.model = model
        self.dim = (1, n_samps)
        self.n_samps = n_samps
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.audio_file = audio_file
        self.n_channels = 1
        self.sample_rate = sample_rate
        # For the E2E model, need to return channels last?
        if channels_last:
            self.expand_axis = 2
        else:
            self.expand_axis = 1

        fs, data = wavfile.read(audio_file)
        if fs != self.sample_rate:
            ratio = self.sample_rate / fs
            resamp = samplerate.resample(data, ratio, "sinc_best")
            print(
                f"Resampling from {self.sample_rate} to {sample_rate}, "
                f"ratio: {ratio}. Had {len(data)} samples, now {len(resamp)}"
            )
            data = resamp
            data = data / 32767
        self.data = data

        # set up list of IDs from data files
        n_points = math.floor(len(self.data) / n_samps)
        print(f"Got {n_points} chunks of audio from {len(self.data)} samples")

        self.list_IDs = range(n_points)

        print(f"Number of examples in dataset: {len(self.list_IDs)}")
        slice: int = 0
        if last > 0.0:
            slice = int(n_points * (1 - last))
            self.list_IDs = self.list_IDs[slice:]
            print(f"Taking Last N points: {len(self.list_IDs)}")
        elif first > 0.0:
            slice = int(n_points * first)
            self.list_IDs = self.list_IDs[:slice]
            print(f"Taking First N points: {len(self.list_IDs)}")

        self.on_epoch_end()

    def get_audio_length(self):
        return self.n_samps

    def __len__(self):
        "Denotes the number of batches per epoch"
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        "Generate one batch of data"
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size : (index + 1) * self.batch_size]
        # Generate data
        X, y = self.__data_generation(indexes)
        return X, y

    def on_epoch_end(self):
        "Updates indexes after each epoch"
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def get_label_size(self):
        res = self.make_prediction(0)
        print(f"Result: {res}")
        return len(res)

    def get_audio(self, index):
        """
        Returns a list of values for the appropriate index
        """
        start = index * self.n_samps
        end = (index + 1) * self.n_samps
        # print(f"Getting audio from {start} to {end}")
        return self.data[start:end]

    def get_parameters(self, audio):
        """
        Takes a list of audio samples, and returns the model's prediction for
        parameters
        """
        return self.model.predict(audio)

    # Make this one cached?
    @lru_cache(maxsize=150000)
    def make_prediction(self, index):
        dat = self.get_audio(index)
        xdat = np.expand_dims(np.vstack([dat]), axis=2)
        pred = self.get_parameters(xdat)
        # print(f"Data: {dat}, Pred: {pred}")
        return pred[0]

    def __data_generation(self, list_IDs_temp):
        # X : (n_samples, *dim, n_channels)
        "Generates data containing batch_size samples"
        # Generate data
        X = []
        y = []
        for i in list_IDs_temp:
            labels = self.make_prediction(i)
            # print(f"Got labels: {labels.shape}")
            y.append(labels)
            X.append(labels)
        # At the moment, just returning all the labels
        Xd = np.vstack(X)
        yd = np.vstack(y)

        return Xd, yd
