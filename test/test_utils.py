import re
from models.utils import utils

class TestUtils:
    def test_load_audio(self):
        assert utils.load_audio is not None

    def test_fingerprint(self):
        result = utils.fingerprint

        assert re.search(r'\.|\s|\\|\:', result) is None
