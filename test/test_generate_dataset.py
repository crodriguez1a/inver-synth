import os
from models.generate_dataset import collect_paths, raw_dataset

class TestGenerateDataset:
    def test_collect_paths(self):
        base_path = os.getcwd() + f'/audio/m1pianoEXS24/M1piano/'
        paths = collect_paths(base_path)

        assert len(paths) == 16

    def test_raw_dataset(self):
        base_path = os.getcwd() + f'/audio/m1pianoEXS24/M1piano/'
        paths = collect_paths(base_path)
        dataset = raw_dataset(paths)

        assert dataset.shape == (15, 1, 16384)
