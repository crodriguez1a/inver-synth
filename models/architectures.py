from dataclasses import dataclass

# Model architectures
@dataclass
class C:
    filters: int
    window_size: tuple
    strides: tuple
    activation: str = 'relu'

"""Conv 1 (2 Layers)"""
c1: C = C(38, (13, 26), (13,26))
c1_layers: list = [c1]

"""Conv 2 (3 Layers)"""
c2_layers : list = [
    C(35, (6,7), (5,6)),
    C(87, (6,9), (5,8))
]

"""Conv 3 (4 Layers)"""
c3_layers: list = [
    C(32, (4,5), (3,4)),
    C(98, (4,6), (3,5)),
    C(128, (4,6), (3,5))
]

"""Conv 4 (5 Layers)"""
c4_layers: list = [
    C(32, (3,4), (2,3)),
    C(65, (3,4), (2,3)),
    C(105, (3,4), (2,3)),
    C(128, (4,5), (3,4))
]

"""Conv 5 (6 Layers)"""
c5_layers: list = [
    C(32, (3,3), (2,2)),
    C(98, (3,3), (2,2)),
    C(128, (3,4), (2,3)),
    C(128, (3,5), (2,4)),
    C(128, (3,3), (2,2))
]

"""Conv 6 (7 Layers)"""
c6_layers: list = [
    C(32, (3,3), (2,2)),
    C(71, (3,3), (2,2)),
    C(128, (3,4), (2,3)),
    C(128, (3,3), (2,2)),
    C(128, (3,3), (2,2)),
    C(128, (3,3), (1,2))
]

"""Conv 6XL, 7 Layers"""
c6XL_layers: list = [
    C(64, (3,3), (2,2)),
    C(128, (3,3), (2,2)),
    C(128, (3,4), (2,3)),
    C(128, (3,3), (2,2)),
    C(256, (3,3), (2,2)),
    C(256, (3,3), (1,2))
]

# @paper:
# These four layers are 1D strided
# convolutional layers that operates on the time axis only.

"""Conv E2E, 11 Layers"""
# @paper:
# The first four layers degenerates to
# 1D strided convolutions by setting
# both K1 and S1 to 1. C(F,K1,K2,S1,S2)
cE2E_1d_layers: list = [
    C(96, (64), (4)),
    C(96, (32), (4)),
    C(128, (16), (4)),
    C(257, (8), (4))
]

# @paper:
# followed by additional six 2D strided convolutional layers that
# are identical to those of Conv6 model
cE2E_2d_layers: list = [
    C(32, (3,3), (2,2)),
    C(71, (3,3), (2,2)),
    C(128, (3,4), (2,3)),
    C(128, (3,3), (2,2)),
    # TODO: fix negative dimensions
    # C(128, (3,3), (2,2)),
    # C(128, (3,3), (1,2))
]

layers_map: dict = {
    'C1': c1_layers,
    'C2': c2_layers,
    'C3': c3_layers,
    'C4': c4_layers,
    'C5': c5_layers,
    'C6': c6_layers,
    'C6XL': c6XL_layers,
    'CE2E': cE2E_1d_layers,
    'CE2E_2D': cE2E_2d_layers,
}
