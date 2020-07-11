import numpy as np

from models.e2e_cnn import get_model

def test_get_model():
    assert get_model('e2e', 16384, 256) is not None
