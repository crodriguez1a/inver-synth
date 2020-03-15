import os
import keras
import logging
import numpy as np
from pathlib import Path
from dotenv import load_dotenv

"""Dotenv Config"""
env_path = Path('.') / '.env'
load_dotenv(dotenv_path=env_path)

"""Audio Samples"""
# NOTE: May need true FM synthesizer generator
# credit: https://beatproduction.net/korg-m1-piano-samples/
m1_sample: str = os.getcwd() + '/audio/samples/M1_Piano_C4.wav'
# credit: https://freewavesamples.com/yamaha-dx7-bass-c2
dx7_sample: str = os.getcwd() + '/audio/samples/Yamaha-DX7-Bass-C2.wav'

"""Data Utils"""


def train_val_split(x_train: np.ndarray,
                    y_train: np.ndarray,
                    split: float = .2,) -> tuple:

    slice: int = int(x_train.shape[0] * split)

    x_val: np.ndarray = x_train[-slice:]
    y_val: np.ndarray = y_train[-slice:]

    x_train = x_train[:-slice]
    y_train = y_train[:-slice]

    return (x_val, y_val, x_train, y_train)


"""Model Utils"""


def summarize_compile(model: keras.Model):
    model.summary(line_length=80, positions=[.33, .65, .8, 1.])
    # Specify the training configuration (optimizer, loss, metrics)
    model.compile(optimizer=keras.optimizers.Adam(),  # Optimizer- Adam [14] optimizer
                  # Loss function to minimize
                  # @paper: Therefore, we converged on using sigmoid activations with binary cross entropy loss.
                  loss=keras.losses.BinaryCrossentropy(),
                  # List of metrics to monitor
                  metrics=[  # @paper: 1) Mean Percentile Rank?
        keras.metrics.MeanAbsolutePercentageError(),
        # @paper: 2) Top-k mean accuracy based evaluation
        # TODO: keras.metrics.TopKCategoricalAccuracy(),
        # https://github.com/tensorflow/tensorflow/issues/9243
        # @paper: 3) Mean Absolute Error based evaluation
        keras.metrics.MeanAbsoluteError(), ])


def fit(model: keras.Model,
        x_train: np.ndarray, y_train: np.ndarray,
        x_val: np.ndarray, y_val: np.ndarray,
        batch_size: int = 16, epochs: int = 100,) -> keras.Model:

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
            data_format: str,
            use_multiprocessing: bool = True,
            verbose: int = 1,) -> np.ndarray:

    result: np.ndarray = model.predict(x=x,
                                       verbose=verbose,
                                       use_multiprocessing=use_multiprocessing)

    # TODO: rearrange for `channels_first/last`
    # is_channels_first: bool = data_format == 'channels_first'
    # result = result[0, 0] if is_channels_first else result[0, :, :, 0]
    # result = result[0, 0]

    return result


def data_format_audio(audio: np.ndarray, data_format: str) -> np.ndarray:
    # `(None, n_channel, n_freq, n_time)` if `'channels_first'`,
    # `(None, n_freq, n_time, n_channel)` if `'channels_last'`,

    if data_format == 'channels_last':
        audio = audio[np.newaxis, :, np.newaxis]
    else:
        audio = audio[np.newaxis, np.newaxis, :]

    return audio
