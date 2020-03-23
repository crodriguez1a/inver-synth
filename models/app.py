import os
from tensorflow import keras
# import keras
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



def compare(target,prediction,params,precision=1,print_output=False):
    if print_output and len(prediction) < 10:
        print(prediction)
        print("Pred: {}".format(np.round(prediction,decimals=2)))
        print("PRnd: {}".format(np.round(prediction)))
        print("Act : {}".format(target))
        print("+" * 5)

    pred:List[ParamValue] = params.decode(prediction)
    act:List[ParamValue] = params.decode(target)
    pred_index:List[int] = [np.array(p.encoding).argmax() for p in pred]
    act_index:List[int] = [np.array(p.encoding).argmax() for p in act]
    width = 8
    names =     "Parameter: "
    act_s =     "Actual:    "
    pred_s =    "Predicted: "
    pred_i =    "Pred. Indx:"
    act_i =     "Act. Index:"
    diff_i =    "Index Diff:"
    for p in act:
        names += p.name.rjust(width)[:width]
        act_s += f'{p.value:>8.2f}'
    for p in pred:
        pred_s += f'{p.value:>8.2f}'
    for p in pred_index:
        pred_i += f'{p:>8}'
    for p in act_index:
        act_i += f'{p:>8}'
    for i in range(len(act_index)):
        diff = pred_index[i] - act_index[i]
        diff_i += f'{diff:>8}'
    exact = 0.0
    close = 0.0
    n_params = len(pred_index)
    for i in range(n_params):
        if pred_index[i] == act_index[i]:
            exact = exact + 1.0
        if abs(pred_index[i] - act_index[i]) <= precision:
            close = close + 1.0
    exact_ratio = exact / n_params
    close_ratio = close / n_params
    if print_output:
        print(names)
        print(act_s)
        print(pred_s)
        print(act_i)
        print(pred_i)
        print(diff_i)
        print("-" * 30)
    return exact_ratio, close_ratio

def evaluate(prediction: np.ndarray,
             x: np.ndarray,
             y: np.ndarray,
             params: ParameterSet,):

    print("Prediction Shape: {}".format(prediction.shape))

    num: int = x.shape[0]
    correct: int = 0
    correct_r: float = 0.0
    close_r: float = 0.0
    for i in range(num):
        should_print = i < 5
        exact,close = compare(target=y[i],prediction=prediction[i],params=params,print_output=should_print)
        if exact == 1.0:
            correct = correct + 1
        correct_r += exact
        close_r += close
    summary = params.explain()
    print("{} Parameters with {} levels (fixed: {})".format(
        summary['n_variable'],summary['levels'],summary['n_fixed']))
    print("Got {} out of {} ({:.1f}% perfect); Exact params: {:.1f}%, Close params: {:.1f}%".format(
        correct,num, correct/num * 100, correct_r/num * 100, close_r/num * 100 ))


def data_format_audio(audio: np.ndarray, data_format: str) -> np.ndarray:
    # `(None, n_channel, n_freq, n_time)` if `'channels_first'`,
    # `(None, n_freq, n_time, n_channel)` if `'channels_last'`,

    if data_format == 'channels_last':
        audio = audio[np.newaxis, :, np.newaxis]
    else:
        audio = audio[np.newaxis, np.newaxis, :]

    return audio
