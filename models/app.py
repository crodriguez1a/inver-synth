import os
from tensorflow import keras
import tensorflow as tf
import tensorflow.keras.backend as K

# import keras
import logging
import numpy as np
from pathlib import Path
from dotenv import load_dotenv
import pandas as pd
from pickle import load


from typing import Dict, Tuple, Sequence, List, Callable
from generators.generator import *
from models.common.data_generator import SoundDataGenerator

from keras.callbacks import CSVLogger



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

def mean_percentile_rank(y_true, y_pred, k=5):
    """
    @paper
    The first evaluation measure is the Mean Percentile Rank
    (MPR) which is computed per synthesizer parameter.
    """
    # TODO
    pass

def top_k_mean_accuracy(y_true, y_pred, k=5):
    """
    @ paper
    The top-k mean accuracy is obtained by computing the top-k
    accuracy for each test example and then taking the mean across
    all examples. In the same manner as done in the MPR analysis,
    we compute the top-k mean accuracy per synthesizer
    parameter for 𝑘 = 1, ... ,5.
    """
    # TODO: per parameter?

    original_shape = tf.shape(y_true)
    y_true = tf.reshape(y_true, (-1, tf.shape(y_true)[-1]))
    y_pred = tf.reshape(y_pred, (-1, tf.shape(y_pred)[-1]))
    top_k = K.in_top_k(y_pred, tf.cast(tf.argmax(y_true, axis=-1), 'int32'), k)
    correct_pred =  tf.reshape(top_k, original_shape[:-1])
    return tf.reduce_mean(tf.cast(correct_pred, tf.float32))


def summarize_compile(model: keras.Model):
    model.summary(line_length=80, positions=[.33, .65, .8, 1.])
    # Specify the training configuration (optimizer, loss, metrics)
    model.compile(optimizer=keras.optimizers.Adam(),  # Optimizer- Adam [14] optimizer
                  # Loss function to minimize
                  # @paper: Therefore, we converged on using sigmoid activations with binary cross entropy loss.
                  loss=keras.losses.BinaryCrossentropy(),
                  # List of metrics to monitor
                  metrics=[
                      # @paper: 1) Mean Percentile Rank?
                      # mean_percentile_rank,
                      # @paper: 2) Top-k mean accuracy based evaluation
                      top_k_mean_accuracy,
                      # @paper: 3) Mean Absolute Error based evaluation
                      keras.metrics.MeanAbsoluteError()
                  ])


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


def compare(target, prediction, params, precision=1, print_output=False):
    if print_output and len(prediction) < 10:
        print(prediction)
        print("Pred: {}".format(np.round(prediction, decimals=2)))
        print("PRnd: {}".format(np.round(prediction)))
        print("Act : {}".format(target))
        print("+" * 5)

    pred: List[ParamValue] = params.decode(prediction)
    act: List[ParamValue] = params.decode(target)
    pred_index: List[int] = [np.array(p.encoding).argmax() for p in pred]
    act_index: List[int] = [np.array(p.encoding).argmax() for p in act]
    width = 8
    names = "Parameter: "
    act_s = "Actual:    "
    pred_s = "Predicted: "
    pred_i = "Pred. Indx:"
    act_i = "Act. Index:"
    diff_i = "Index Diff:"
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
        exact, close = compare(
            target=y[i], prediction=prediction[i], params=params, print_output=should_print)
        if exact == 1.0:
            correct = correct + 1
        correct_r += exact
        close_r += close
    summary = params.explain()
    print("{} Parameters with {} levels (fixed: {})".format(
        summary['n_variable'], summary['levels'], summary['n_fixed']))
    print("Got {} out of {} ({:.1f}% perfect); Exact params: {:.1f}%, Close params: {:.1f}%".format(
        correct, num, correct/num * 100, correct_r/num * 100, close_r/num * 100))


def data_format_audio(audio: np.ndarray, data_format: str) -> np.ndarray:
    # `(None, n_channel, n_freq, n_time)` if `'channels_first'`,
    # `(None, n_freq, n_time, n_channel)` if `'channels_last'`,

    if data_format == 'channels_last':
        audio = audio[np.newaxis, :, np.newaxis]
    else:
        audio = audio[np.newaxis, np.newaxis, :]

    return audio


"""
Wrap up the whole training process in a standard function. Gets a callback
to actually make the model, to keep it as flexible as possible.
# Params:
# - dataset_name (dataset name)
# - model_name: (C1..C6,e2e)
# - model_callback: function taking name,inputs,outputs,data_format and returning a Keras model
# - epochs: int
# - dataset_dir: place to find input data
# - output_dir: place to put outputs
# - parameters_file (override parameters filename)
# - dataset_file (override dataset filename)
# - data_format (channels_first or channels_last)
# - run_name: to save this run as
"""


def train_model(
        # Main options
        dataset_name: str, model_name: str, epochs: int, model_callback: Callable[[str, int, int, str], keras.Model],
        dataset_dir: str, output_dir: str,  # Directory names
        dataset_file: str = None, parameters_file: str = None,
        run_name: str = None,
        data_format: str = 'channels_last',
        save_best: bool = True,
        resume:bool = False,
        checkpoint:bool=True):

    if not dataset_file:
        dataset_file = os.getcwd() + "/" + dataset_dir + "/" + \
            dataset_name + "_data.hdf5"
    if not parameters_file:
        parameters_file = os.getcwd() + "/" + dataset_dir + "/" + \
            dataset_name + "_params.pckl"
    if not run_name:
        run_name = dataset_name + "_" + model_name

    model_file = f"{output_dir}/{run_name}.h5"
    best_model_file = f"{output_dir}/{run_name}_best.h5"
    checkpoint_model_file = f"{output_dir}/{run_name}_checkpoint.h5"
    history_file = f"{output_dir}/{run_name}.csv"
    history_graph_file = f"{output_dir}/{run_name}.pdf"

    gpu_avail = tf.test.is_gpu_available()  # True/False
    cuda_gpu_avail = tf.test.is_gpu_available(cuda_only=True)  # True/False

    print("+"*30)
    print(f"++ {run_name}")
    print(
        f"Running model: {model_name} on dataset {dataset_file} (parameters {parameters_file}) for {epochs} epochs")
    print(f"Saving model in {output_dir} as {model_file}")
    print(f"Saving history as {history_file}")
    print(f"GPU: {gpu_avail}, with CUDA: {cuda_gpu_avail}")
    print("+"*30)

    os.makedirs(output_dir, exist_ok=True)

    # Get training and validation generators
    params = {'data_file': dataset_file, 'batch_size': 64, 'shuffle': True}
    training_generator = SoundDataGenerator(first=0.8, **params)
    validation_generator = SoundDataGenerator(last=0.2, **params)
    n_samples = training_generator.get_audio_length()
    print(f"get_audio_length: {n_samples}")
    n_outputs = training_generator.get_label_size()

    # set keras image_data_format
    # NOTE: on CPU only `channels_last` is supported
    keras.backend.set_image_data_format(data_format)

    model : keras.Model = None
    if resume and os.path.exists(checkpoint_model_file):
        history = pd.read_csv(history_file)
        # Note - its zero indexed in the file, but 1 indexed in the display
        initial_epoch:int = max(history.iloc[:,0]) + 1
        print(f"Resuming from model file: {checkpoint_model_file} after epoch {initial_epoch}")
        model = keras.models.load_model(checkpoint_model_file,custom_objects={
            'top_k_mean_accuracy':top_k_mean_accuracy
        })
    else:
        model = model_callback(model_name=model_name,
                                            inputs=n_samples,
                                            outputs=n_outputs,
                                            data_format=data_format)
        # Summarize and compile the model
        summarize_compile(model)
        initial_epoch = 0
        open(history_file, 'w').close()

    callbacks = []
    best_callback = keras.callbacks.ModelCheckpoint(filepath=best_model_file,
                                                  save_weights_only=False,
                                                  save_best_only=True,
                                                  verbose=1)
    checkpoint_callback = keras.callbacks.ModelCheckpoint(filepath=checkpoint_model_file,
                                                  save_weights_only=False,
                                                  save_best_only=False,
                                                  verbose=1)
    if save_best:
        callbacks.append(best_callback)
    if checkpoint:
        callbacks.append(checkpoint_callback)
    callbacks.append( CSVLogger(history_file, append=True) )
    # Fit the model
    history = model.fit(x=training_generator,
                        validation_data=validation_generator,
                        epochs=epochs, callbacks=callbacks,
                        initial_epoch=initial_epoch)

    # Save model
    model.save(model_file)

    # Save history
    try:
        hist_df = pd.DataFrame(history.history)
        try:
            fig = hist_df.plot(subplots=True,figsize=(8,25))
            fig[0].get_figure().savefig(history_graph_file)
        except Exception as e:
            print("Couldn't create history graph")
            print(e)

    except Exception as e:
        print("Couldn't save history")
        print(e)



    # evaluate prediction on random sample from validation set
    # Parameter data - needed for decoding!
    with open(parameters_file, 'rb') as f:
        parameters: ParameterSet = load(f)

    # Shuffle data
    validation_generator.on_epoch_end()
    X, y = validation_generator.__getitem__(0)
    prediction: np.ndarray = model.predict(X)
    evaluate(prediction, X, y, parameters)
