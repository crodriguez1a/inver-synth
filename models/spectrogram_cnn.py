import os

import numpy as np

from tensorflow import keras
from kapre.time_frequency import Spectrogram

from models.app import summarize_compile, fit, data_format_audio, train_val_split, evaluate
from models.common.utils import utils
from models.common.architectures import layers_map
from models.common.data_generator import SoundDataGenerator

from generators.generator import *
from pickle import load


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
                   n_outputs:int,
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
        n_outputs, activation='sigmoid', name='predictions')(x)

    return keras.Model(inputs=inputs, outputs=outputs)

if __name__ == "__main__":


    # TEMP!

    dataset: str = os.getcwd() + "/" + os.getenv('TRAINING_SET')
    params = {
            'data_file': dataset,
            'batch_size': 64,
            'shuffle': True
            }

    training_generator = SoundDataGenerator(first=0.8, **params)
    validation_generator = SoundDataGenerator(last=0.2, **params)

    n_samples = training_generator.get_audio_length()
    print(f"get_audio_length: {n_samples}")
    n_outputs = training_generator.get_label_size()

    # Load in training data
    # x_train: np.ndarray = np.load(dataset)
    # n_samples: int = x_train.shape[2]
    # n_examples: int = x_train.shape[0]
    # print("Length: {}, number of examples: {}".format(n_samples,n_examples))

    # Load in label data
    # labels: str = os.getcwd() + os.getenv('LABELS')
    # y_train: np.ndarray = np.load(labels)
    # n_labels: int = y_train.shape[1]
    # n_label_examples: int = y_train.shape[0]
    # print("Label Length: {}, number of examples: {}".format(n_labels,n_label_examples))


    # Parameter data - needed for decoding!
    param_file: str = os.getcwd() + "/" + os.getenv('PARAMETERS')
    with open(param_file,'rb') as f:
        parameters : ParameterSet = load(f)

    # set keras image_data_format
    # NOTE: on CPU only `channels_last` is supported
    data_format: str = os.getenv('IMAGE_DATA_FORMAT', 'channels_last')
    keras.backend.set_image_data_format(data_format)

    arch_layers = layers_map.get(os.getenv('ARCHITECTURE', 'C1'))


    model: keras.Model = assemble_model(np.zeros([1,n_samples]),
                                        n_outputs=n_outputs,
                                        arch_layers=arch_layers,
                                        data_format=data_format)

    # Reserve samples for validation
    #split: float = .2
    #x_val, y_val, x_train, y_train = train_val_split(x_train, y_train, split)
    #print("Shapes: x_val={}, y_val={}, x_train={}, y_train={}".format(x_val.shape,y_val.shape,x_train.shape,y_train.shape))

    # Summarize and compile the model
    summarize_compile(model)

    # Fit, with validation
    epochs: int = int(os.getenv('EPOCHS', 100))  # @paper: 100

    # NOTE: `fit_generator` trains the model on data generated
    # batch-by-batch using `training_generator` (keras.utils.Sequence instance)
    model.fit_generator(generator=training_generator,
                        validation_data=validation_generator,
                        epochs=epochs,)

    # model: keras.Model = fit(model,
                             #x_train, y_train,
                             #x_val, y_val,
                             #epochs=epochs,)

    # evaluate prediction on random sample from validation set
    validation_generator.on_epoch_end()
    X,y = validation_generator.__getitem__(0)
    prediction: np.ndarray = model.predict(X)
    evaluate(prediction, X, y, parameters)
