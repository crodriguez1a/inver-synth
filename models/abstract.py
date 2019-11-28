from dataclasses import dataclass

@dataclass
class C:
    filters: int
    window_size: tuple
    strides: tuple
    activation: str = 'relu'
