import numpy as np

# import keras
from kapre.time_frequency import Spectrogram
from tensorflow import keras

from generators.generator import *
from models.app import (
    data_format_audio,
    evaluate,
    fit,
    summarize_compile,
    train_val_split,
)
from models.common.architectures import layers_map
from models.common.data_generator import SoundDataGenerator
from models.common.utils import utils

"""
The STFT spectrogram of the input signal is fed
into a 2D CNN that predicts the synthesizer parameter
configuration. This configuration is then used to produce
a sound that is similar to the input sound.
"""


"""Model Architecture"""
# @ paper:
# 1 2D Strided Convolution Layer C(38,13,26,13,26)
# where C(F,K1,K2,S1,S2) stands for a ReLU activated
# 2D strided convolutional layer with F filters in size of (K1,K2)
# and strides (S1,S2).


def assemble_model(
    src: np.ndarray,
    n_outputs: int,
    arch_layers: list,
    n_dft: int = 512,  # Orig:128
    n_hop: int = 256,  #  Orig:64
    data_format: str = "channels_first",
) -> keras.Model:

    inputs = keras.Input(shape=src.shape, name="stft")

    # @paper: Spectrogram based CNN that receives the (log) spectrogram matrix as input

    # @kapre:
    # abs(Spectrogram) in a shape of 2D data, i.e.,
    # `(None, n_channel, n_freq, n_time)` if `'channels_first'`,
    # `(None, n_freq, n_time, n_channel)` if `'channels_last'`,
    x = Spectrogram(
        n_dft=n_dft,
        n_hop=n_hop,
        input_shape=src.shape,
        trainable_kernel=True,
        name="static_stft",
        image_data_format=data_format,
        return_decibel_spectrogram=True,
    )(inputs)

    # Swaps order to match the paper?
    # TODO: dig in to this (GPU only?)
    if data_format == "channels_first":  # n_channel, n_freq, n_time)
        x = keras.layers.Permute((1, 3, 2))(x)
    else:
        x = keras.layers.Permute((2, 1, 3))(x)

    for arch_layer in arch_layers:
        x = keras.layers.Conv2D(
            arch_layer.filters,
            arch_layer.window_size,
            strides=arch_layer.strides,
            activation=arch_layer.activation,
            data_format=data_format,
        )(x)

    # Flatten down to a single dimension
    x = keras.layers.Flatten()(x)

    # @paper: sigmoid activations with binary cross entropy loss
    # @paper: FC-512
    x = keras.layers.Dense(512)(x)

    # @paper: FC-368(sigmoid)
    outputs = keras.layers.Dense(n_outputs, activation="sigmoid", name="predictions")(x)

    return keras.Model(inputs=inputs, outputs=outputs)


"""
Standard callback to get a model ready to train
"""


def get_model(
    model_name: str, inputs: int, outputs: int, data_format: str = "channels_last"
) -> keras.Model:
    arch_layers = layers_map.get("C1")
    if model_name in layers_map:
        arch_layers = layers_map.get(model_name)
    else:
        print(
            f"Warning: {model_name} is not compatible with the spectrogram model. C1 Architecture will be used instead."
        )
    return assemble_model(
        np.zeros([1, inputs]),
        n_outputs=outputs,
        arch_layers=arch_layers,
        data_format=data_format,
    )


if __name__ == "__main__":

    from models.app import train_model
    from models.runner import standard_run_parser

    # Get a standard parser, and the arguments out of it
    parser = standard_run_parser()
    args = parser.parse_args()
    setup = vars(args)

    # distinguish model type for reshaping
    setup["model_type"] = "STFT"

    # Actually train the model
    train_model(model_callback=get_model, **setup)
