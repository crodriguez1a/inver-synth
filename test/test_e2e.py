import numpy as np
from models.e2e_cnn import assemble_model
from models.common.architectures import cE2E_1d_layers, cE2E_2d_layers


class TestModel_E2E_CNN:
    def test_output_shape(self):
        input_2d = np.load("test/sample_input.npy")
        labels = np.random.uniform((1, 12))
        n_labels = labels.shape[0]
        model = assemble_model(
            input_2d,
            n_labels,
            cE2E_1d_layers,
            cE2E_2d_layers,
            data_format="channels_first",
        )

        # @paper
        # All models end with an output layer of
        # 368 with sigmoid activations (to match ℎ’s dimension).

        assert n_labels in model.layers[-1].output_shape
        assert len(model.layers) == 15
