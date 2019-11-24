import logging
import os
from dataclasses import dataclass, field

import numpy as np
import keras

import librosa
import soundfile as sf
from kapre.time_frequency import Spectrogram

"""
The STFT spectrogram of the input signal is fed
into a 2D CNN that predicts the synthesizer parameter
configuration. This configuration is then used to produce
a sound that is similar to the input sound.
"""

def input_sound(path: str, sr: int=16384, duration: float=1.) -> tuple:
    # signal in a duration of 1 second with a sampling rate of 16384Hz
    # Input (16384 raw audio)
    y_audio, sample_rate = librosa.load(path,
                                        sr=sr, # `None` preserves sample rate
                                        duration=duration,)
    return (y_audio, sample_rate)

def inverse_stft():
    # tf.signal.inverse_stft
    pass

def stft_audio(S: np.ndarray) -> np.ndarray:
    # Inverse STFT to audio
    return librosa.griffinlim(S)

def validate_mae():
    # true features vs output features
    # mean absolute error per synthesizer parameter
    pass

# Sequence a Spectrogram based CNN that receives the (log) spectrogram matrix as input
def summarize_compile(model: keras.Model):
    model.summary(line_length=80, positions=[.33, .65, .8, 1.])
    # Specify the training configuration (optimizer, loss, metrics)
    model.compile(optimizer=keras.optimizers.Adam(), # Optimizer- Adam [14] optimizer
                  # Loss function to minimize
                  # V. Therefore, we converged on using sigmoid activations with binary cross entropy loss.
                  loss=keras.losses.BinaryCrossentropy(),
                  # List of metrics to monitor
                  metrics=[ # 1) Mean Percentile Rank?
                            keras.metrics.MeanAbsolutePercentageError(),
                            # 2) Top-k mean accuracy based evaluation
                            # TODO metrics.TopKCategoricalAccuracy(),
                            # 3) Mean Absolute Error based evaluation
                            keras.metrics.MeanAbsoluteError(),])

def fit(model: keras.Model,
        x_train: np.ndarray,
        y_train: np.ndarray,
        batch_size:int=16,
        epochs:int=1,) -> keras.Model:


    # with a minibatch size of 16 for
    # 100 epochs. The best weights for each model were set by
    # employing an early stopping procedure.
    logging.info('# Fit model on training data')
    history = model.fit(x_train, y_train,
                        batch_size=batch_size,
                        epochs=epochs,)
                        # Early stopping procedure:
                        #   We pass some validation for
                        #   monitoring validation loss and metrics
                        #   at the end of each epoch
                        # TODO validation_data=(x_val, y_val))

    # The returned "history" object holds a record
    # of the loss values and metric values during training
    logging.info('\nhistory dict:', history.history)

    return model

def predict(model: keras.Model,
            audio_src: np.ndarray,
            logam: bool=False,) -> np.ndarray:

    # prepare batch
    num_chnls, wv = model.input_shape[1:]
    src: np.ndarray = audio_src[:wv]
    src_batch: np.ndarray = src[np.newaxis, np.newaxis, :]

    # predict
    result: np.ndarray = model.predict(x=src_batch)

    # shuffle for `channels first`
    if keras.backend.image_data_format == 'channels_first':
        result = result[0, 0]
    else:
        result = result[0, :, :, 0]

    return result

"""Conv 1 - 2 Layers"""

# 1 2D Strided Convolution Layer C(38,13,26,13,26)
# where C(F,K1,K2,S1,S2) stands for a ReLU activated
# 2D strided convolutional layer with F filters in size of (K1,K2)
# and strides (S1,S2).

@dataclass
class C1:
    filters: int
    window_size: tuple
    strides: tuple
    activation: str = 'relu'

c1: C1 = C1(38, (13, 26), (13,26))

def assemble_c1_model(src: np.ndarray) -> keras.Model:
    inputs = keras.Input(shape=src.shape, name='stft')

    # TODO parametrize
    n_dft: int = 512
    n_hop: int = 64

    x = Spectrogram(n_dft=n_dft, n_hop=n_hop, input_shape=src.shape,
                    return_decibel_spectrogram=True, power_spectrogram=2.0,
                    trainable_kernel=False, name='static_stft',
                    image_data_format='channels_first',)(inputs)

    x = keras.layers.Conv2D(c1.filters,
                            c1.window_size,
                            strides=c1.strides,
                            activation=c1.activation,
                            data_format='channels_first',)(x) # data_format=None, dilation_rate=(1, 1), activation=None, use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None)

    # sigmoid activations with binary cross entropy loss
    # FC-512
    x = keras.layers.Dense(512)(x)
    # FC-368(sigmoid)
    outputs = keras.layers.Dense(368, activation='sigmoid', name='predictions')(x)
    model = keras.Model(inputs=inputs, outputs=outputs)

    return model

"""Conv 6XL, 7 Layers"""
# TODO

if __name__ == "__main__":
    # load audio sample

    # NOTE: May need true FM synthesizer generator
    # credit: https://beatproduction.net/korg-m1-piano-samples/
    m1_sample: str = os.getcwd() + '/audio/samples/M1_Piano_C4.wav'
    # credit: https://freewavesamples.com/yamaha-dx7-bass-c2
    dx7_sample: str = os.getcwd() + '/audio/samples/Yamaha-DX7-Bass-C2.wav'

    input_audio_path: str = os.getenv('AUDIO_WAV_INPUT', dx7_sample)
    duration: float = 1
    y_audio, sample_rate = input_sound(input_audio_path, duration=duration)

    # assemble
    model_template: keras.Model = assemble_c1_model(np.expand_dims(y_audio, axis=0))

    batch_input_shape = (2,) + model_template.input_shape[1:]
    batch_output_shape = (2,) + model_template.output_shape[1:]

    x_train = np.random.uniform(size=batch_input_shape)
    y_train = np.random.uniform(size=batch_output_shape)

    # Reserve samples for validation
    # slice = 12
    # x_val = x_train[-slice:]
    # y_val = y_train[-slice:]
    # x_train = x_train[:-slice]
    # y_train = y_train[:-slice]

    summarize_compile(model_template)

    model: keras.Model = fit(model_template,
                             x_train,
                             y_train,
                             epochs=10)

    result: np.ndarray = predict(model, y_audio, sample_rate)

    # new_audio = stft_audio(result)
    # sf.write('audio/new_audio.wav', new_audio, sample_rate)
