import numpy as np

from models.spectrogram_cnn import get_model


def test_get_model():
    assert get_model("C1", 16384, 256) is not None
