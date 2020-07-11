![Build](https://github.com/crodriguez1a/inver-synth/workflows/Build/badge.svg?branch=master)

# inver-synth
> A Python implementation of the [InverSynth](https://arxiv.org/abs/1812.06349) method (Barkan, Tsiris, Koenigstein, Katz)

---

NOTE: This implementation is a **work in progress**. Contributions are welcome.

## Installation

```
poetry shell
poetry install
```

## Generating a Training Set of Simple Sinusoidal Synthesis

Create the following output directories:

```
test_waves
test_datasets
```

Then run:
```
python -m generators.fm_generator
```

This will generate a dataset of simple sinusoidal synthesis, which has:

- training data in `test_datasets/`
- wave files in `test_waves/`


### Experimenting with the E2E & Spectrogram models

First, assign values to following environment variables in a `.env`:

Required variables:

```
--model {C1,C2,C3,C4,C5,C6,C6XL,e2e}
--dataset_name DATASET_NAME
```

Optional variables:

```
--epochs EPOCHS
--dataset_dir DATASET_DIR
--output_dir OUTPUT_DIR
--dataset_file DATASET_FILE
--parameters_file PARAMETERS_FILE
--data_format {channels_last,channels_first}
--run_name RUN_NAME
```

Selecting an architecture (default is `C1`):

- `C1`, `C2`, `C3`, `C4`, `C5`, `C6`, `C6XL`, `CE2E`, `CE2E_2D`

![workflow](docs/img/architectures.png "Mimimun, Maximum")

Training the models:

>  End-to-End learning. A CNN predicts the synthesizer parameter configuration directly from the raw audio. The first
convolutional layers perform 1D convolutions that learn an alternative representation for the STFT Spectrogram. Then, a
stack of 2D convolutional layers analyze the learned representation to predict the synthesizer parameter configuration.

```
python -m models.e2e_cnn --model C3 --dataset_name inversynth_full
```

or

>  The STFT spectrogram of the input signal is fed into a 2D CNN that predicts the
synthesizer parameter configuration. This configuration is then used to produce a sound that is similar to the input sound.

```
python -m models.spectrogram_cnn --model C5 --dataset_name inversynth
```

### Datasets

See [GENERATING.md](GENERATING.md)
