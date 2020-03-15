import os

import numpy as np
import librosa

import keras  # TODO: update to tf.keras when kapre goes to tf2.0
# https://github.com/keunwoochoi/kapre/pull/58/commits/a3268110471466e4799621d0ae39bd05d84ee275
from kapre.time_frequency import Spectrogram

from models.app import summarize_compile, fit, predict, data_format_audio, train_val_split
from models.common.utils import utils
from models.common.architectures import layers_map

"""
The STFT spectrogram of the input signal is fed
into a 2D CNN that predicts the synthesizer parameter
configuration. This configuration is then used to produce
a sound that is similar to the input sound.
"""

"""Audio Pre-processing"""


def input_raw_audio(path: str, sr: int = 16384, duration: float = 1.) -> tuple:
    # @paper: signal in a duration of 1 second with a sampling rate of 16384Hz
    # @paper: Input (16384 raw audio)
    return utils.load_audio(path, sr, duration)


"""Model Architecture"""
# @ paper:
# 1 2D Strided Convolution Layer C(38,13,26,13,26)
# where C(F,K1,K2,S1,S2) stands for a ReLU activated
# 2D strided convolutional layer with F filters in size of (K1,K2)
# and strides (S1,S2).


def assemble_model(src: np.ndarray,
                   arch_layers: list,
                   n_dft: int = 128,
                   n_hop: int = 64,
                   data_format: str = 'channels_first',) -> keras.Model:

    inputs = keras.Input(shape=src.shape, name='stft')

    # @paper: Spectrogram based CNN that receives the (log) spectrogram matrix as input

    # @kapre:
    # abs(Spectrogram) in a shape of 2D data, i.e.,
    # `(None, n_channel, n_freq, n_time)` if `'channels_first'`,
    # `(None, n_freq, n_time, n_channel)` if `'channels_last'`,
    x: Spectrogram = Spectrogram(n_dft=n_dft, n_hop=n_hop, input_shape=src.shape,
                                 trainable_kernel=True, name='static_stft',
                                 image_data_format=data_format,
                                 return_decibel_spectrogram=True,)(inputs)

    for arch_layer in arch_layers:
        x = keras.layers.Conv2D(arch_layer.filters,
                                arch_layer.window_size,
                                strides=arch_layer.strides,
                                activation=arch_layer.activation,
                                data_format=data_format,)(x)

    # Flatten down to a single dimension
    x = keras.layers.Flatten()(x)
    # @paper: sigmoid activations with binary cross entropy loss
    # @paper: FC-512
    x = keras.layers.Dense(512)(x)

    # @paper: FC-368(sigmoid)
    outputs = keras.layers.Dense(
        368, activation='sigmoid', name='predictions')(x)

    return keras.Model(inputs=inputs, outputs=outputs)


if __name__ == "__main__":
    # Load audio sample
    input_audio_path: str = os.getenv('AUDIO_WAV_INPUT')
    # Define audio sample max duration
    duration: float = 1
    # Extract raw audio
    y_audio, sample_rate = input_raw_audio(input_audio_path, duration=duration)

    # set keras image_data_format
    data_format: str = os.getenv('IMAGE_DATA_FORMAT', 'channels_first')
    keras.backend.set_image_data_format(data_format)

    # input should be a 2D array, `(audio_channel, audio_length)`.
    input_2d: np.ndarray = y_audio[np.newaxis, :]

    arch_layers = layers_map.get(os.getenv('ARCHITECTURE', 'C1'))
    model: keras.Model = assemble_model(input_2d,
                                        arch_layers=arch_layers,
                                        data_format=data_format)

    # n-synth bass dataset https://magenta.tensorflow.org/datasets/nsynth#files
    dataset: str = os.getcwd() + os.getenv('TRAINING_SET')
    x_train: np.ndarray = np.load(dataset)
    n_samples: int = x_train.shape[0]

    # generate labels arbitrarily
    y_train: np.ndarray = np.random.uniform(
        size=(n_samples,) + model.output_shape[1:])

    # Reserve samples for validation
    split: float = .2
    x_val, y_val, x_train, y_train = train_val_split(x_train, y_train, split)

    # Summarize and compile the model
    summarize_compile(model)

    # Fit, with validation
    epochs: int = int(os.getenv('EPOCHS', 100))  # @paper: 100
    model: keras.Model = fit(model,
                             x_train, y_train,
                             x_val, y_val,
                             epochs=epochs,)

    if os.getenv('EXPERIMENTATION', False):
        # Predict
        x_test: np.ndarray = data_format_audio(y_audio, data_format)
        result: np.ndarray = predict(model, x_test, data_format)

        # Save model
        save_path: str = os.getenv('SAVED_MODELS_PATH')
        utils.h5_save(model, save_path)

        # Write audio
        new_audio: np.ndarray = utils.stft_to_audio(result)
        wav_out: str = os.getenv('AUDIO_WAV_OUTPUT')
        librosa.output.write_wav(wav_out, new_audio, sample_rate)
