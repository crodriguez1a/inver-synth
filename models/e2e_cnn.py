import logging
import os

import numpy as np
import librosa

import keras # TODO: update to tf.keras when kapre goes to tf2.0
# https://github.com/keunwoochoi/kapre/pull/58/commits/a3268110471466e4799621d0ae39bd05d84ee275
from kapre.time_frequency import Spectrogram

from models.app import summarize_compile, fit, predict, data_format_audio, train_val_split
from models.common.utils import utils
from models.common.architectures import *

"""
End-to-End learning. A CNN predicts the synthesizer
parameter configuration directly from the raw audio.
The first convolutional layers perform 1D convolutions
that learn an alternative representation for the STFT
Spectrogram. Then, a stack of 2D convolutional layers
analyze the learned representation to predict the
synthesizer parameter configuration.
"""

"""Audio Pre-processing"""
def input_raw_audio(path: str, sr: int=16384, duration: float=1.) -> tuple:
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
                   c1d_layers: list,
                   c2d_layers: list,
                   n_dft: int=128,
                   n_hop: int=64,
                   data_format: str='channels_first',) -> keras.Model:

    # define shape not including batch size
    inputs = keras.Input(shape=src.shape, name='raw_audio')

    # @paper:
    # The first four layers degenerates to
    # 1D strided convolutions by setting
    # both K1 and S1 to 1. C(F,K1,K2,S1,S2)
    for i, arch_layer in enumerate(c1d_layers):
        x = inputs if i == 0 else x
        x = keras.layers.Conv1D(arch_layer.filters,
                                arch_layer.window_size,
                                strides=arch_layer.strides,
                                data_format=data_format,)(x)

    # @paper: learned representation
    x = keras.layers.Lambda(lambda x: keras.backend.expand_dims(x, axis=1))(x)

    # @paper:
    # followed by additional six 2D strided convolutional layers that
    # are identical to those of Conv6 model
    for arch_layer in c2d_layers:
        x = keras.layers.Conv2D(arch_layer.filters,
                                arch_layer.window_size,
                                strides=arch_layer.strides,
                                activation=arch_layer.activation,
                                data_format=data_format,)(x)

    # @paper: sigmoid activations with binary cross entropy loss

    # @paper: FC-512
    x = keras.layers.Dense(512)(x)

    # @paper: FC-368(sigmoid)
    outputs = keras.layers.Dense(368, activation='sigmoid', name='predictions')(x)

    return keras.Model(inputs=inputs, outputs=outputs)


if __name__ == "__main__":
    # Load audio sample
    input_audio_path: str = os.getenv('AUDIO_WAV_INPUT')
    # Define audio sample max duration
    duration: float = 1
    # Extract raw audio
    y_audio, sample_rate = input_raw_audio(input_audio_path, duration=duration)

    # `channels_first` = 1 channel, 1 sample of a signal with length n
    input_2d: np.ndarray = y_audio[np.newaxis, :]

    # set keras image_data_format
    data_format: str = 'channels_first'
    keras.backend.set_image_data_format(data_format)

    model: keras.Model = assemble_model(input_2d,
                                        cE2E_1d_layers,
                                        cE2E_2d_layers,
                                        data_format=data_format,)

    dataset: str = os.getcwd() + os.getenv('TRAINING_SET')
    x_train: np.ndarray = np.load(dataset)
    n_samples: int = x_train.shape[0]

    y_train: np.ndarray = np.random.uniform(size=(n_samples,) + model.output_shape[1:])

    # Reserve samples for validation
    split: float = .2
    x_val, y_val, x_train, y_train = train_val_split(x_train, y_train, split)

    # Summarize and compile the model
    summarize_compile(model)

    # Fit, with validation
    epochs: int = int(os.getenv('EPOCHS', '100')) # @paper: 100
    model: keras.Model = fit(model,
                             x_train, y_train,
                             x_val, y_val,
                             epochs=epochs,)

    if os.getenv('EXPERIMENTATION', False):
        # `channels_first` = 1 channel, 1 sample of a signal with length n
        x_test: np.ndarray = data_format_audio(y_audio, data_format)
        result: np.ndarray = predict(model, x_test, data_format)

        # Save model
        save_path: str = os.getenv('SAVED_MODELS_PATH')
        if save_path:
            utils.h5_save(model, save_path, filename_attrs=f'n_epochs={epochs}')

        # Write audio
        new_audio: np.ndarray = utils.stft_to_audio(result)
        wav_out: str = os.getenv('AUDIO_WAV_OUTPUT')
        if wav_out:
            librosa.output.write_wav(wav_out, new_audio, sample_rate)
