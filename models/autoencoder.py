from keras.layers import Input, Dense
from keras.models import Model
from dataclasses import dataclass
from tensorflow import keras
from typing import List

from models.common.data_generator import SoundDataGenerator
from models.common.soundfile_generator import SoundfileGenerator
from models.app import top_k_mean_accuracy
import tensorflowjs as tfjs


"""
This is a start a doing an autoencoder, in order to learn simplified
representations of synthesis parameters. Code based on:
https://blog.keras.io/building-autoencoders-in-keras.html
"""

# this is the size of our encoded representations
input_size = 16834


@dataclass
class Autoencoder:
    """
    Simple utility class to pass round all the bits of an Autoencoder
    """

    autoencoder: Model
    encoder: Model
    decoder: Model

    def save(self, base_filename, model_name):
        self.autoencoder.save(base_filename.format(model_name, "autoencoder") + ".h5")
        self.encoder.save(base_filename.format(model_name, "encoder") + ".h5")
        self.decoder.save(base_filename.format(model_name, "decoder") + ".h5")
        ae_js_fn = base_filename.format(model_name, "autoencoder")
        print(f"Saving AE to : {ae_js_fn}")
        tfjs.converters.save_keras_model(self.autoencoder, ae_js_fn)
        en_js_fn = base_filename.format(model_name, "encoder")
        print(f"Saving Enc to : {en_js_fn}")
        tfjs.converters.save_keras_model(self.encoder, en_js_fn)
        de_js_fn = base_filename.format(model_name, "decoder")
        print(f"Saving Dec to : {de_js_fn}")
        tfjs.converters.save_keras_model(self.decoder, de_js_fn)


def create_model(input_size: int = 300, encoding_size: int = 10):
    """
    Create an autoencoder model of the given shape
    """
    # this is our input placeholder
    input_img = Input(shape=(input_size,))
    # "encoded" is the encoded representation of the input
    encoded = Dense(encoding_size, activation="relu")(input_img)
    # "decoded" is the lossy reconstruction of the input
    decoded = Dense(input_size, activation="sigmoid")(encoded)

    # this model maps an input to its reconstruction
    autoencoder = Model(input_img, decoded)

    # this model maps an input to its encoded representation
    encoder = Model(input_img, encoded)

    # create a placeholder for an encoded (32-dimensional) input
    encoded_input = Input(shape=(encoding_size,))
    # retrieve the last layer of the autoencoder model
    decoder_layer = autoencoder.layers[-1]
    # create the decoder model
    decoder = Model(encoded_input, decoder_layer(encoded_input))

    autoencoder.summary(line_length=80, positions=[0.33, 0.65, 0.8, 1.0])
    # The values in the blog train poorly; we use the values from the paper
    # See app.py where they are used in main model training
    autoencoder.compile(
        optimizer=keras.optimizers.Adam(), loss=keras.losses.BinaryCrossentropy()
    )
    return Autoencoder(autoencoder, encoder, decoder)


def train_on_parameters(
    ae: Autoencoder,
    training_generator: Model,
    validation_generator: Model,
    epochs: int = 50,
):
    """
    Train the autoencoder on a set of known parameters
    """
    ae.autoencoder.fit(
        x=training_generator, validation_data=validation_generator, epochs=epochs
    )


def train_on_audio(ae: Autoencoder, model: Model, audio: List[str]):
    """
    Train the autoencoder on an audio input
    """


def run_database(model_name):
    print("Training autoencoder")
    # All this should be from command line args
    encoding_size = 35
    epochs = 50
    dataset_file = f"./test_datasets/{model_name}_data.hdf5"

    # Get training and validation generators
    params = {
        "data_file": dataset_file,
        "batch_size": 64,
        "shuffle": True,
        "for_autoencoder": True,
    }
    training_generator = SoundDataGenerator(first=0.8, **params)
    validation_generator = SoundDataGenerator(last=0.2, **params)
    n_outputs = training_generator.get_label_size()
    print(f"Label size: {n_outputs}")

    ae = create_model(input_size=n_outputs, encoding_size=encoding_size)

    train_on_parameters(
        ae,
        training_generator=training_generator,
        validation_generator=validation_generator,
        epochs=epochs,
    )
    ae.save("./output/{}_{}", model_name=model_name)


def run_audio_file(model_file, audio_file, output_name):
    print(f"Training autoencoder on {audio_file} using model {model_file}")
    # All this should be from command line args
    encoding_size = 5
    epochs = 50

    print("Loading model...")
    model = keras.models.load_model(
        model_file, custom_objects={"top_k_mean_accuracy": top_k_mean_accuracy}
    )
    print("Model loaded")

    # Get training and validation generators
    params = {
        "audio_file": audio_file,
        "batch_size": 64,
        "shuffle": True,
        "model": model,
    }
    training_generator = SoundfileGenerator(first=0.8, **params)
    validation_generator = SoundfileGenerator(last=0.2, **params)
    n_outputs = training_generator.get_label_size()

    print(f"Creating Autoencoder, size = {encoding_size}")
    ae = create_model(input_size=n_outputs, encoding_size=encoding_size)
    print("Autoencoder created... training...")

    train_on_parameters(
        ae,
        training_generator=training_generator,
        validation_generator=validation_generator,
        epochs=epochs,
    )
    ae.save("./output/{}_{}", model_name=output_name)


if __name__ == "__main__":
    # run_database( "inversynth_small")
    audio_file = "./test_waves/ShortExample.wav"
    model_file = "./output/dexed_2osc_e2e_best.h5"
    model_file = "./output/lokomotiv_full_e2e_best.h5"
    run_audio_file(model_file, audio_file, "lokomotiv_atcha")
