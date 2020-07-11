import os
from tensorflow import keras
from generators.generator import *
import pickle
from scipy.io import wavfile
from scipy.io.wavfile import write as write_wav
import re


"""
This module generates comparisons - takes the original sound + params,
then generates a file with the predicted parameters
"""


def compare(
    model: keras.Model,
    generator: SoundGenerator,
    parameters: ParameterSet,
    orig_file: str,
    output_dir: str,
    orig_params,
    length: float,
    sample_rate: int,
    extra: dict = {},
):
    # (copy original file if given)
    base_filename = orig_file.replace(".wav", "")
    base_filename = re.sub(r".*/", "", base_filename)
    copy_file: str = f"{output_dir}/{base_filename}_copy.wav"
    regen_file: str = f"{output_dir}/{base_filename}_duplicate.wav"
    reconstruct_file: str = f"{output_dir}/{base_filename}_reconstruct.wav"
    print(f"Creating copy as {copy_file}")

    # Load the wave file
    fs, data = wavfile.read(orig_file)
    # Copy original file to make sure
    write_wav(copy_file, sample_rate, data)

    # Decode original params, and regenerate output (make sure its correct)
    orig = parameters.encoding_to_settings(orig_params)
    generator.generate(orig, regen_file, length, sample_rate, extra)

    # Run the wavefile into the model for prediction
    X = [data]
    Xd = np.expand_dims(np.vstack(X), axis=2)
    # Get encoded parameters out of model
    result = model.predict(Xd)[0]

    # Decode prediction, and reconstruct output
    predicted = parameters.encoding_to_settings(result)
    generator.generate(predicted, reconstruct_file, length, sample_rate, extra)


def run_comparison(
    model: keras.Model,
    generator: SoundGenerator,
    run_name: str,
    indices=None,
    num_samples=10,
    data_dir="./test_datasets",
    output_dir="./comparison",
    length=1.0,
    sample_rate=16384,
    shuffle=True,
    extra={},
):
    # Figure out data file and params file from run name
    data_file = f"{data_dir}/{run_name}_data.hdf5"
    parameters_file = f"{data_dir}/{run_name}_params.pckl"
    print(f"Reading parameters from {parameters_file}")
    parameters = pickle.load(open(parameters_file, "rb"))

    output_dir = f"{output_dir}/{run_name}/"
    os.makedirs(output_dir, exist_ok=True)

    database = h5py.File(data_file, "r")

    if not indices:
        ids = np.array(range(len(database["files"])))
        if shuffle:
            np.random.shuffle(ids)
        indices = ids[0:num_samples]

    # filename
    for i in indices:
        print("Looking at index: {}".format(i))
        filename = database["files"][i]
        labels = database["labels"][i]
        compare(
            model=model,
            generator=generator,
            parameters=parameters,
            orig_file=filename,
            output_dir=output_dir,
            orig_params=labels,
            length=length,
            sample_rate=sample_rate,
            extra=extra,
        )
    # Generate


if __name__ == "__main__":

    note_length = 0.8
    sample_rate = 16384

    lokomotiv = True
    fm = True

    if lokomotiv:
        from generators.vst_generator import *

        run_name = "lokomotiv_full"
        model_file = "output/lokomotiv_full_e2e_best.h5"
        plugin = "/Library/Audio/Plug-Ins/VST/Lokomotiv.vst"
        config_file = "plugin_config/lokomotiv.json"
        generator = VSTGenerator(vst=plugin, sample_rate=sample_rate)
        with open(config_file, "r") as f:
            config = json.load(f)

        model = keras.models.load_model(model_file)
        run_comparison(
            model,
            generator,
            run_name,
            num_samples=100,
            extra={"note_length": note_length, "config": config},
        )

    if fm:
        from generators.fm_generator import *

        run_name = "inversynth_full"
        model_file = "output/inversynth_full_e2e_best.h5"
        generator = InverSynthGenerator()
        model = keras.models.load_model(model_file)
        run_comparison(model, generator, run_name, num_samples=100)
