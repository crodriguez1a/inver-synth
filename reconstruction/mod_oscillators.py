import itertools
import random
import sys
from abc import ABC, abstractmethod
from math import cos, fabs, floor, log, pi, sin, sqrt
from typing import Generator, Iterator, List, Optional, Sequence, Tuple

from synthplayer import params
from synthplayer.oscillators import *


class ModSine(Oscillator):
    """Sine Wave oscillator."""
    def __init__(self, frequency: Oscillator, amplitude: Oscillator, phase: float = 0.0, bias: float = 0.0,
                 fm_lfo: Optional[Oscillator] = None, samplerate: int = 0) -> None:
        # The FM compensates for the phase change by means of phase_correction.
        # See http://stackoverflow.com/questions/3089832/sine-wave-glissando-from-one-pitch-to-another-in-numpy
        # and http://stackoverflow.com/questions/28185219/generating-vibrato-sine-wave
        # The same idea is applied to the other waveforms to correct their phase with FM.
        super().__init__(samplerate)
        self.frequency = frequency.blocks()
        self.amplitude = amplitude.blocks()
        self.bias = bias
        self.fm = fm_lfo.blocks() if fm_lfo else Linear(0.0).blocks()
        self._phase = phase

    def blocks(self) -> Generator[List[float], None, None]:
        phase_correction = self._phase*2*pi
        freq_previous = 0.0
        increment = 2.0*pi/self.samplerate
        t = 0.0
        # optimizations:
        bias = self.bias
        frequency = self.frequency
        amplitude = self.amplitude
        while True:
            block = []  # type: List[float]
            fm_block = next(self.fm)
            freq_block = next(self.frequency)
            amp_block = next(self.amplitude)
            for i in range(params.norm_osc_blocksize):
                freq = freq_block[i]*(1.0+fm_block[i])
                phase_correction += (freq_previous-freq)*t
                freq_previous = freq
                ampl = amp_block[i]
                val = sin(t*freq+phase_correction)*amp_block[i]+bias
                block.append(val)
                t += increment
            yield block


class ModTriangle(Oscillator):
    """Perfect triangle wave oscillator (not using harmonics)."""
    def __init__(self, frequency: Oscillator, amplitude: Oscillator, phase: float = 0.0, bias: float = 0.0,
                 fm_lfo: Optional[Oscillator] = None, samplerate: int = 0) -> None:
        super().__init__(samplerate)
        self.frequency = frequency.blocks()
        self.amplitude = amplitude.blocks()
        self.bias = bias
        self.fm = fm_lfo.blocks() if fm_lfo else Linear(0.0).blocks()
        self._phase = phase

    def blocks(self) -> Generator[List[float], None, None]:
        phase_correction = self._phase
        freq_previous = 0.0
        increment = 1.0/self.samplerate
        t = 0.0
        # optimizations:
        frequency = self.frequency
        amplitude = self.amplitude
        bias = self.bias
        while True:
            block = []  # type: List[float]
            fm_block = next(self.fm)
            freq_block = next(self.frequency)
            amp_block = next(self.amplitude)
            for i in range(params.norm_osc_blocksize):
                freq = freq_block[i] * (1.0+fm_block[i])
                phase_correction += (freq_previous-freq)*t
                freq_previous = freq
                tt = t*freq+phase_correction
                block.append(4.0*amp_block[i]*(fabs((tt+0.75) % 1.0 - 0.5)-0.25)+bias)
                t += increment
            yield block


class ModSquare(Oscillator):
    """Perfect square wave [max/-max] oscillator (not using harmonics)."""
    def __init__(self, frequency: Oscillator, amplitude: Oscillator, phase: float = 0.0, bias: float = 0.0,
                 fm_lfo: Optional[Oscillator] = None, samplerate: int = 0) -> None:
        super().__init__(samplerate)
        self.frequency = frequency.blocks()
        self.amplitude = amplitude.blocks()
        self.bias = bias
        self.fm = fm_lfo.blocks() if fm_lfo else Linear(0.0).blocks()
        self._phase = phase

    def blocks(self) -> Generator[List[float], None, None]:
        phase_correction = self._phase
        freq_previous = 0.0
        increment = 1.0/self.samplerate
        t = 0.0
        # optimizations:
        frequency = self.frequency
        amplitude = self.amplitude
        bias = self.bias
        while True:
            block = []  # type: List[float]
            fm_block = next(self.fm)
            freq_block = next(self.frequency)
            amp_block = next(self.amplitude)
            for i in range(params.norm_osc_blocksize):
                freq = freq_block[i]*(1.0+fm_block[i])
                phase_correction += (freq_previous-freq)*t
                freq_previous = freq
                tt = t*freq + phase_correction
                block.append((-amp_block[i] if int(tt*2) % 2 else amp_block[i])+bias)
                t += increment
            yield block


class ModSawtooth(Oscillator):
    """Perfect sawtooth waveform oscillator (not using harmonics)."""
    def __init__(self, frequency: Oscillator, amplitude: Oscillator, phase: float = 0.0, bias: float = 0.0,
                 fm_lfo: Optional[Oscillator] = None, samplerate: int = 0) -> None:
        super().__init__(samplerate)
        self.frequency = frequency.blocks()
        self.amplitude = amplitude.blocks()
        self.bias = bias
        self.fm = fm_lfo.blocks() if fm_lfo else Linear(0.0).blocks()
        self._phase = phase

    def blocks(self) -> Generator[List[float], None, None]:
        increment = 1.0/self.samplerate
        freq_previous = 0.0
        phase_correction = self._phase
        t = 0.0
        # optimizations:
        frequency = self.frequency
        amplitude = self.amplitude
        bias = self.bias
        while True:
            block = []  # type: List[float]
            fm_block = next(self.fm)
            freq_block = next(self.frequency)
            amp_block = next(self.amplitude)

            for i in range(params.norm_osc_blocksize):
                freq = freq_block[i]*(1.0+fm_block[i])
                phase_correction += (freq_previous-freq)*t
                freq_previous = freq
                tt = t*freq + phase_correction
                block.append(bias+amp_block[i]*2.0*(tt - floor(0.5+tt)))
                t += increment
            yield block
