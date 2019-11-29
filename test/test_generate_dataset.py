import os
from models.generate_dataset import collect_paths, raw_dataset

class TestGenerateDataset:
    def test_collect_paths(self):
        base_path = os.getcwd() + f'/audio/samples/'
        paths = collect_paths(base_path)

        assert len(paths) == 3

    def test_raw_dataset(self):
        base_path = os.getcwd() + f'/audio/samples/'
        paths = collect_paths(base_path)
        dataset = raw_dataset(paths)

        assert dataset.shape == (3, 1, 16384)
