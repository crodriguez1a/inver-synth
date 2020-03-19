import os
from tensorflow import keras
import logging
import numpy as np
from pathlib import Path
from dotenv import load_dotenv

from typing import Dict, Tuple, Sequence, List
from generators.generator import *

"""Dotenv Config"""
env_path = Path('.') / '.env'
load_dotenv(dotenv_path=env_path)


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
        keras.metrics.MeanAbsoluteError(),
        keras.metrics.CategoricalAccuracy(),])


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


def _prediction_shape(prediction: np.ndarray,
                      x: np.ndarray,
                      y: np.ndarray,
                      params: ParameterSet,):
    print("Prediction Shape: {}".format(prediction.shape))
    for i in range(min(x.shape[0],30)):
        print("Pred: {}".format(np.round(prediction[i],decimals=2)))
        print("PRnd: {}".format(np.round(prediction[i])))
        print("Act : {}".format(y[i]))
        print("+" * 5)
        names = ""
        act_s = ""
        pred_s = ""
        pred:List[ParamValue] = params.decode(prediction[i])
        act:List[ParamValue] = params.decode(y[i])
        for p in act:
            names += p.name.rjust(10)
            act_s += f'{p.value:>10}'
        for p in pred:
            pred_s += f'{p.value:>10}'
        print(names)
        print(act_s)
        print(pred_s)
        print("-" * 30)

def evaluate(prediction: np.ndarray,
             x: np.ndarray,
             y: np.ndarray,
             params: ParameterSet,):

    _prediction_shape(prediction, x, y, params)

    num: int = x.shape[0]
    correct: int = 0
    for i in range(num):
        if np.absolute( np.round(prediction[i]) - y[i] ).sum() < 0.1:
            correct = correct + 1
    print("Got {} out of {} ({}%)".format(correct,num, correct/num * 100 ))


def data_format_audio(audio: np.ndarray, data_format: str) -> np.ndarray:
    # `(None, n_channel, n_freq, n_time)` if `'channels_first'`,
    # `(None, n_freq, n_time, n_channel)` if `'channels_last'`,

    if data_format == 'channels_last':
        audio = audio[np.newaxis, :, np.newaxis]
    else:
        audio = audio[np.newaxis, np.newaxis, :]

    return audio
