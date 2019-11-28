import logging
import os

import numpy as np
import librosa

import keras # TODO: update to tf.keras when kapre goes to tf2.0
# https://github.com/keunwoochoi/kapre/pull/58/commits/a3268110471466e4799621d0ae39bd05d84ee275
from kapre.time_frequency import Spectrogram

from models.utils import utils
from models.abstract import C

"""
The STFT spectrogram of the input signal is fed
into a 2D CNN that predicts the synthesizer parameter
configuration. This configuration is then used to produce
a sound that is similar to the input sound.
"""

"""Audio Samples"""
# NOTE: May need true FM synthesizer generator
# credit: https://beatproduction.net/korg-m1-piano-samples/
m1_sample: str = os.getcwd() + '/audio/samples/M1_Piano_C4.wav'
# credit: https://freewavesamples.com/yamaha-dx7-bass-c2
dx7_sample: str = os.getcwd() + '/audio/samples/Yamaha-DX7-Bass-C2.wav'

"""Audio Pre-processing"""
def input_raw_audio(path: str, sr: int=16384, duration: float=1.) -> tuple:
    # @paper: signal in a duration of 1 second with a sampling rate of 16384Hz
    # @paper: Input (16384 raw audio)
    return utils.load_audio(path, sr, duration)

def stft_to_audio(S: np.ndarray) -> np.ndarray:
    # Inverse STFT to audio
    # tf.signal.inverse_stft
    return librosa.griffinlim(S)

"""Model Utils"""
def summarize_compile(model: keras.Model):
    model.summary(line_length=80, positions=[.33, .65, .8, 1.])
    # Specify the training configuration (optimizer, loss, metrics)
    model.compile(optimizer=keras.optimizers.Adam(), # Optimizer- Adam [14] optimizer
                  # Loss function to minimize
                  # @paper: Therefore, we converged on using sigmoid activations with binary cross entropy loss.
                  loss=keras.losses.BinaryCrossentropy(),
                  # List of metrics to monitor
                  metrics=[ # @paper: 1) Mean Percentile Rank?
                            keras.metrics.MeanAbsolutePercentageError(),
                            # @paper: 2) Top-k mean accuracy based evaluation
                            # TODO: keras.metrics.TopKCategoricalAccuracy(),
                            # https://github.com/tensorflow/tensorflow/issues/9243
                            # @paper: 3) Mean Absolute Error based evaluation
                            keras.metrics.MeanAbsoluteError(),])

def fit(model: keras.Model,
        x_train: np.ndarray, y_train: np.ndarray,
        x_val: np.ndarray, y_val: np.ndarray,
        batch_size:int=16, epochs:int=100,) -> keras.Model:

    # @paper:
    # with a minibatch size of 16 for
    # 100 epochs. The best weights for each model were set by
    # employing an early stopping procedure.
    logging.info('# Fit model on training data')
    history = model.fit(x_train, y_train,
                        batch_size=batch_size,
                        epochs=epochs,
                        # @paper:
                        # Early stopping procedure:
                        # We pass some validation for
                        # monitoring validation loss and metrics
                        # at the end of each epoch
                        validation_data=(x_val, y_val),)

    # The returned "history" object holds a record
    # of the loss values and metric values during training
    logging.info('\nhistory dict:', history.history)

    return model

def predict(model: keras.Model,
            x: np.ndarray,
            logam: bool=False,) -> np.ndarray:

    # predict
    result: np.ndarray = model.predict(x=x)

    # rearrange for `channels first`
    is_channels_first: bool = keras.backend.image_data_format == 'channels_first'
    result = result[0, 0] if is_channels_first else result[0, :, :, 0]

    return result

"""Model Architecture"""
# @ paper:
# 1 2D Strided Convolution Layer C(38,13,26,13,26)
# where C(F,K1,K2,S1,S2) stands for a ReLU activated
# 2D strided convolutional layer with F filters in size of (K1,K2)
# and strides (S1,S2).

def assemble_model(src: np.ndarray,
                   arch_layers: list,
                   n_dft: int=128,
                   n_hop: int=64,) -> keras.Model:

    inputs = keras.Input(shape=src.shape, name='stft')

    # @paper: Spectrogram based CNN that receives the (log) spectrogram matrix as input

    # @kapre:
    # abs(Spectrogram) in a shape of 2D data, i.e.,
    # `(None, n_channel, n_freq, n_time)` if `'channels_first'`,
    # `(None, n_freq, n_time, n_channel)` if `'channels_last'`,
    x: Spectrogram = Spectrogram(n_dft=n_dft, n_hop=n_hop, input_shape=src.shape,
                                 trainable_kernel=True, name='static_stft',
                                 image_data_format='channels_first',
                                 return_decibel_spectrogram=True,)(inputs)
    
    for arch_layer in arch_layers:
        x = keras.layers.Conv2D(arch_layer.filters,
                                arch_layer.window_size,
                                strides=arch_layer.strides,
                                activation=arch_layer.activation,
                                data_format='channels_first',)(x) # data_format=None, dilation_rate=(1, 1), activation=None, use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None)

    # @paper: sigmoid activations with binary cross entropy loss

    # @paper: FC-512
    x = keras.layers.Dense(512)(x)

    # @paper: FC-368(sigmoid)
    outputs = keras.layers.Dense(368, activation='sigmoid', name='predictions')(x)

    return keras.Model(inputs=inputs, outputs=outputs)


if __name__ == "__main__":
    # Load audio sample
    input_audio_path: str = os.getenv('AUDIO_WAV_INPUT', dx7_sample)
    # Define audio sample max duration
    duration: float = 1
    # Extract raw audio
    y_audio, sample_rate = input_raw_audio(input_audio_path, duration=duration)

    # Model assembly
    """Conv 1 (2 Layers)"""
    c1: C = C(38, (13, 26), (13,26))
    c1_layers: list = [c1]

    """Conv 2 (3 Layers)"""
    c3_layers : list = [
        C(35, (6,7), (5,6)),
        C(87, (6,9), (5,8))
    ]

    """Conv 3 (4 Layers)"""
    c4_layers: list = [
        C(32, (4,5), (3,4)),
        C(98, (4,6), (3,5)),
        C(128, (4,6), (3,5))
    ]

    """Conv 6XL, 7 Layers"""
    c6XL_layers: list = [
        C(64, (3,3), (2,2)),
        C(128, (3,3), (2,2)),
        C(128, (3,4), (2,3)),
        C(128, (3,3), (2,2)),
        C(256, (3,3), (2,2)),
        C(256, (3,3), (1,2))
    ]

    # `channels_first` input should be a 2D array, `(audio_channel, audio_length)`.
    input_2d: np.ndarray = y_audio[np.newaxis, :]

    model: keras.Model = assemble_model(input_2d,
                                        arch_layers=c3_layers,)

    # n-synth bass dataset https://magenta.tensorflow.org/datasets/nsynth#files
    nsynth_bass_dataset: str = os.getcwd() + '/data/large/dataset_2019-11-26_10:14:08_880147.npy'
    x_train: np.ndarray = np.load(os.getenv('TRAINING_SET', nsynth_bass_dataset))
    n_samples: int = x_train.shape[0]

    y_train: np.ndarray = np.random.uniform(size=(n_samples,) + model.output_shape[1:])

    # Reserve samples for validation
    slice = int(n_samples * .2)
    x_val = x_train[-slice:]
    y_val = y_train[-slice:]
    x_train = x_train[:-slice]
    y_train = y_train[:-slice]

    # Summarize and compile the model
    summarize_compile(model)

    # Fit, with validation
    epochs: int = 3 #100
    model: keras.Model = fit(model,
                             x_train, y_train,
                             x_val, y_val,
                             epochs=epochs,)

    # TEMP
    if os.getenv('EXPERIMENTATION', True): # TODO
        # Predict
        # `channels_first` = 1 channel, 1 sample of a signal with length n
        x_test: np.ndarray = y_audio[np.newaxis, np.newaxis, :]
        result: np.ndarray = predict(model, x_test, sample_rate)

        # Write audio
        new_audio: np.ndarray = stft_to_audio(result)
        wav_out: str = os.getenv('AUDIO_WAV_OUTPUT', 'audio/outputs/new_audio.wav')
        librosa.output.write_wav(wav_out, new_audio, sample_rate)

        # Save model
        save_path: str = os.getenv('SAVED_MODELS_PATH', '/models/saved/large')
        utils.h5_save(model, save_path, filename_attrs=f'n_epochs={epochs}')
