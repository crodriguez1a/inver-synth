![Build](https://github.com/crodriguez1a/inver-synth/workflows/Build/badge.svg?branch=master)

# inver-synth
> A Python implementation of the [InverSynth](https://arxiv.org/abs/1812.06349) method (Barkan, Tsiris, Koenigstein, Katz)

---

*NOTE: This implementation is a work in progress. Contributions are welcome.*

## Generating a Training Set of Simple Sinusoidal Synthesis

Create the following output directories:

```
test_waves
└── sine
test_datasets
```

Then run:
```
python -m generators.sine_generator
```

This will generate a dataset of simple sinusoidal synthesis, which has:

- training data in `test_datasets/sine_generator_input.npy`
- labels in `/test_datasets/sine_generator_labels.npy`
- wave files in `test_waves/sine/`

Then assign environment variables to point to the output:

```
# .env
TRAINING_SET=/test_datasets/sine_generator_input.npy
LABELS=/test_datasets/sine_generator_labels.npy
```

### Experimenting with the E2E & Spectrogram models

First, assign values to following environment variables in a `.env`:

- `AUDIO_WAV_INPUT` - the input sound to attempt to match

- `TRAINING_SET` - a path to training data

- `EPOCHS` - number of training epochs (default is 100 as prescribed in paper)

- `SAVED_MODELS_PATH` - if set, will save weights to `h5` and architecture to `JSON`

- `AUDIO_WAV_OUTPUT` - if set, will convert prediction output to raw audio and save as a `wav`

- `ARCHITECTURE` - select from pre-defined InverSynth Architectures.

Select one of the following (default is `C1`):

- `C1`, `C2`, `C3`, `C4`, `C5`, `C6`, `C6XL`, `CE2E`, `CE2E_2D`

![workflow](docs/img/architectures.png "Mimimun, Maximum")

Then run the following:

>  End-to-End learning. A CNN predicts the synthesizer parameter configuration directly from the raw audio. The first
convolutional layers perform 1D convolutions that learn an alternative representation for the STFT Spectrogram. Then, a
stack of 2D convolutional layers analyze the learned representation to predict the synthesizer parameter configuration.

```
python -m models.e2e_cnn
```

or

>  The STFT spectrogram of the input signal is fed into a 2D CNN that predicts the
synthesizer parameter configuration. This configuration is then used to produce a sound that is similar to the input sound.

```
python -m models.spectrogram_cnn
```
