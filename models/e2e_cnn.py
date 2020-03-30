import os

import numpy as np

from tensorflow import keras
# import keras

from models.app import summarize_compile, fit, data_format_audio, train_val_split, evaluate
from models.common.utils import utils
from models.common.architectures import cE2E_1d_layers, cE2E_2d_layers

from models.common.data_generator import SoundDataGenerator

from generators.generator import ParameterSet
from pickle import load
import pandas as pd

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
                   n_outputs: int,
                   c1d_layers: list,
                   c2d_layers: list,
                   n_dft: int = 128,
                   n_hop: int = 64,
                   data_format: str = 'channels_first',) -> keras.Model:

    # define shape not including batch size
    inputs = keras.Input(shape=src.shape, name='raw_audio')

    # @paper:
    # The first four layers degenerates to
    # 1D strided convolutions by setting
    # both K1 and S1 to 1. C(F,K1,K2,S1,S2)
    x = None
    for i, arch_layer in enumerate(c1d_layers):
        x = inputs if i == 0 else x
        x = keras.layers.Conv1D(arch_layer.filters,
                                arch_layer.window_size,
                                strides=arch_layer.strides,
                                data_format=data_format,)(x)

    # @paper: learned representation
    # Trying a Reshape instead of Lambda, as it's portable to tfjs
    #x = keras.layers.Lambda(lambda x: keras.backend.expand_dims(x, axis=3))(x)
    x = keras.layers.Reshape((61, 257,1), input_shape=(61,257))(x)


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

    # Flatten down to a single dimension
    x = keras.layers.Flatten()(x)

    # @paper: FC-512
    x = keras.layers.Dense(512)(x)

    # @paper: FC-368(sigmoid)
    outputs = keras.layers.Dense(
        n_outputs, activation='sigmoid', name='predictions')(x)

    return keras.Model(inputs=inputs, outputs=outputs)

def get_model(model_name:str,inputs:int,outputs:int,data_format:str='channels_last')->keras.Model:
    return assemble_model(np.zeros([inputs,1]),
                                        outputs,
                                        cE2E_1d_layers,
                                        cE2E_2d_layers,
                                        data_format=data_format,)

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

    # Parameter data - needed for decoding!
    param_file: str = os.getcwd() + "/" + os.getenv('PARAMETERS')
    with open(param_file,'rb') as f:
        parameters : ParameterSet = load(f)

    # set keras image_data_format
    # NOTE: on CPU only `channels_last` is supported
    data_format: str = os.getenv('IMAGE_DATA_FORMAT','channels_last')
    keras.backend.set_image_data_format(data_format)

    model: keras.Model = assemble_model(np.zeros([n_samples,1]),
                                        n_outputs,
                                        cE2E_1d_layers,
                                        cE2E_2d_layers,
                                        data_format=data_format,)



    # Summarize and compile the model
    summarize_compile(model)

    # Fit, with validation
    epochs: int = int(os.getenv('EPOCHS', 100))  # @paper: 100

    #cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath="best_e2e_model.h5",
                                                 #save_weights_only=False,
                                                 #save_best_only=True,
                                                 #verbose=1)

    # NOTE: `fit_generator` trains the model on data generated
    # batch-by-batch using `training_generator` (keras.utils.Sequence instance)
    history = model.fit(x=training_generator,
                        validation_data=validation_generator,
                        epochs=epochs,)

    hist_df = pd.DataFrame(history.history)
    with open("e2e_training.csv", mode='w') as f:
        hist_df.to_csv(f)
    # evaluate prediction on random sample from validation set
    validation_generator.on_epoch_end()
    X,y = validation_generator.__getitem__(0)
    prediction: np.ndarray = model.predict(X)
    evaluate(prediction, X, y, parameters)
    model.save("trained_e2d_model.h5")

    # Save model
    save_path: str = os.getenv('SAVED_MODELS_PATH', '')
    utils.h5_save(model, save_path)
