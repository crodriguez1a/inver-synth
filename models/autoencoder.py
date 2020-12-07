from keras.layers import Input, Dense
from keras.models import Model
from dataclasses import dataclass
from tensorflow import keras
from typing import List
import h5py as h5

from models.common.data_generator import SoundDataGenerator
from models.common.soundfile_generator import SoundfileGenerator
from models.app import top_k_mean_accuracy
import tensorflowjs as tfjs



'''
This is a start a doing an autoencoder, in order to learn simplified
representations of synthesis parameters. Code based on:
https://blog.keras.io/building-autoencoders-in-keras.html
'''

# this is the size of our encoded representations
input_size = 16834


@dataclass
class Autoencoder():
    '''
    Simple utility class to pass round all the bits of an Autoencoder
    '''
    autoencoder: Model
    encoder: Model
    decoder: Model

    def save(self, base_filename, model_name):
        #h5.get_config().default_file_mode = 'rw'
        ae_js_fn = base_filename.format(model_name, "autoencoder")
        print(f"Saving AE to : {ae_js_fn}")
        self.autoencoder.save(ae_js_fn+".h5")
        tfjs.converters.save_keras_model(self.autoencoder, ae_js_fn)
        en_js_fn = base_filename.format(model_name, "encoder")
        print(f"Saving Enc to : {en_js_fn}")
        self.encoder.save(en_js_fn+".h5")
        tfjs.converters.save_keras_model(self.encoder,en_js_fn)
        de_js_fn = base_filename.format(model_name, "decoder")
        print(f"Saving Dec to : {de_js_fn}")
        self.decoder.save(de_js_fn+".h5")
        tfjs.converters.save_keras_model(self.decoder,de_js_fn)


def create_autoencoder(input_size: int = 300, encoding_size: int = 10):
    '''
    Create an autoencoder model of the given shape
    '''
    # this is our input placeholder
    input_img = keras.Input(shape=(input_size,))
    # "encoded" is the encoded representation of the input
    encoded = keras.layers.Dense(encoding_size, activation='relu')(input_img)
    # "decoded" is the lossy reconstruction of the input
    decoded = keras.layers.Dense(input_size, activation='sigmoid')(encoded)

    # this model maps an input to its reconstruction
    autoencoder = keras.Model(input_img, decoded)

    # this model maps an input to its encoded representation
    encoder = keras.Model(input_img, encoded)

    # create a placeholder for an encoded (32-dimensional) input
    encoded_input = keras.Input(shape=(encoding_size,))
    # retrieve the last layer of the autoencoder model
    decoder_layer = autoencoder.layers[-1]
    # create the decoder model
    decoder = keras.Model(encoded_input, decoder_layer(encoded_input))

    autoencoder.summary(line_length=80, positions=[.33, .65, .8, 1.])
    # The values in the blog train poorly; we use the values from the paper
    # See app.py where they are used in main model training
    autoencoder.compile(optimizer=keras.optimizers.Adam(),
                        loss=keras.losses.BinaryCrossentropy())
    return Autoencoder(autoencoder, encoder, decoder)


def train_on_parameters(ae: Autoencoder,
                        training_generator: Model,
                        validation_generator: Model,
                        epochs: int = 50):
    '''
    Train the autoencoder on a set of known parameters
    '''
    ae.autoencoder.fit(x=training_generator,
                       validation_data=validation_generator,
                       epochs=epochs)


def train_on_audio(ae: Autoencoder, model: Model, audio: List[str]):
    '''
    Train the autoencoder on an audio input
    '''
    pass


def run_database(model_name):
    print("Training autoencoder")
    # All this should be from command line args
    encoding_size = 35
    epochs = 50
    dataset_file = f"./test_datasets/{model_name}_data.hdf5"

    # Get training and validation generators
    params = {
        'data_file': dataset_file,
        'batch_size': 64,
        'shuffle': True,
        'for_autoencoder': True
    }
    training_generator = SoundDataGenerator(first=0.8, **params)
    validation_generator = SoundDataGenerator(last=0.2, **params)
    n_outputs = training_generator.get_label_size()
    print(f"Label size: {n_outputs}")

    ae = create_autoencoder(input_size=n_outputs, encoding_size=encoding_size)

    train_on_parameters(ae,
                        training_generator=training_generator,
                        validation_generator=validation_generator,
                        epochs=epochs)
    ae.save("./output/{}_{}", model_name=model_name)


def train_autoencoder_on_audio(model_name, output_name, audio_file, size=5, output_dir="./output", epochs=100):
    model_file = f"{output_dir}/{model_name}.h5"
    print(f"Training autoencoder on {audio_file} using model {model_file} with {size} dimensions ({epochs} epochs)")

    print("Loading model...")
    model = keras.models.load_model(model_file, custom_objects={
        'top_k_mean_accuracy':top_k_mean_accuracy
    })
    print("Model loaded")

    # Get training and validation generators
    params = {
        'files': audio_file,
        'batch_size': 64,
        'shuffle': True,
        'model':model
    }
    training_generator = SoundfileGenerator(first=0.8, **params)
    validation_generator = SoundfileGenerator(last=0.2, **params)
    n_outputs = training_generator.get_label_size()

    print(f"Creating Autoencoder, i/o size = {n_outputs}, coded size = {size}")
    ae = create_autoencoder(input_size=n_outputs, encoding_size=size)
    print("Autoencoder created... training...")

    train_on_parameters(ae,
                        training_generator=training_generator,
                        validation_generator=validation_generator,
                        epochs=epochs)
    ae.save(output_dir + "/{}_{}", model_name=output_name)


if __name__ == "__main__":
    import argparse
    import glob
    parser = argparse.ArgumentParser(description='''
Train an Autoencoder to give a low dimensional representation of an existing model.
This uses a piece of audio to build a model, so it won't have the full parameter space
but will capture the qualities of the given audio in the reduced feature space.
This means that you can create a preset of the synth with a few knobs, that works
for a particular style or sonic area.

The autoencoder will be saved as the complete model, but also the coder and decoder
separately, so it can be easily used to give a low dimensional feature space.

A typical pipeline might be:
Audio -> Prediction Model -> Coder -> <here's some knobs to twiddle or interpolate> -> Decoder -> Synth

''')
    parser.add_argument('--model', dest='model_name', type=str, required=True,
                        help='Name of existing model to work with (relative to model directory)')
    parser.add_argument('--name', dest='output_name', type=str, required=True,
                        help='Name of created model')
    parser.add_argument('--audio_file', required=True, action='append',
                        help='Path to the audio file to be used for training the autoencoder')
    parser.add_argument('--size', type=int, default=5,
                        help='How many dimensions to have in middle layer')
    parser.add_argument('--epochs', type=int, default=100,
                        help='How many epochs to run')
    parser.add_argument('--model_dir', default='./output',
                        help='Directory to find models and to save output')

    args = parser.parse_args()
    setup = vars(args)
    filenames = []
    for f in args.audio_file:
        filenames.extend(glob.glob(f))
    train_autoencoder_on_audio(model_name=args.model_name,
            output_name=args.output_name,
            audio_file=filenames,
            size=args.size,
            output_dir=args.model_dir,
            epochs=args.epochs)
