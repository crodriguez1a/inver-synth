from models.app import train_model
from models.e2e_cnn import get_model as get_e2e
from models.spectrogram_cnn import get_model as get_spectrogram

import argparse


def standard_run_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Setup and train a model, storing the output"
    )
    parser.add_argument(
        "--model",
        dest="model_name",
        type=str,
        choices=["C1", "C2", "C3", "C4", "C5", "C6", "C6XL", "e2e"],
        default="e2e",
        help="Model architecture to run",
    )
    parser.add_argument(
        "--dataset_name",
        default="InverSynth",
        help='Name of the dataset to use - other filenames are generated from this. If you have a file "modelname_data.hdf5", put in "modelname"',
    )
    parser.add_argument(
        "--epochs", type=int, default=100, help="How many epochs to run"
    )
    parser.add_argument(
        "--dataset_dir",
        default="test_datasets",
        help="Directory full of datasets to use",
    )
    parser.add_argument(
        "--output_dir",
        default="output",
        help="Directory to store the final model and history",
    )
    parser.add_argument(
        "--dataset_file", default=None, help="Specify an exact dataset file to use"
    )
    parser.add_argument(
        "--parameters_file",
        default=None,
        help="Specify an exact parameters file to use",
    )
    parser.add_argument(
        "--data_format",
        type=str,
        choices=["channels_last", "channels_first"],
        default="channels_last",
        help="Image data format for Keras. If CPU only, has to be channels_last",
    )
    parser.add_argument(
        "--run_name",
        type=str,
        dest="run_name",
        help="Name to save the output under. Defaults to dataset_name + model",
    )
    parser.add_argument(
        "--resume",
        dest="resume",
        action="store_const",
        const=True,
        default=False,
        help="Look for a checkpoint file to resume from",
    )
    return parser


if __name__ == "__main__":

    print("Starting model runner")
    # Get a standard parser, and the arguments out of it
    parser = standard_run_parser()
    args = parser.parse_args()
    setup = vars(args)

    print("Parsed arguments")
    # Figure out the model callback
    model_callback = get_spectrogram
    if setup["model_name"] == "e2e":
        model_callback = get_e2e

    # Actually train the model
    train_model(model_callback=model_callback, **setup)
