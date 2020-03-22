from generators.generator import *
import numpy as np
import math

from generators.parameters import freq_range, param_range

from synthplayer.synth import *
from synthplayer.oscillators import *




class SynthplayerGenerator(SoundGenerator):
    def generate(self,parameters:dict,filename:str,length:float,sample_rate:int,extra:dict={})->np.ndarray:
        n_samps = int(length * sample_rate)
        data = []

        osc = self.construct_generator(sample_rate,parameters,extra)
        gen = osc.blocks()
        while len(data) < n_samps:
            data.extend(next(gen))
        return np.array(data)

    """
    Simple example generator - two sine waves, amp and frequency
    """
    def construct_generator(self,sample_rate:int,parameters:dict,extra:dict):
        s1 = Sine(parameters['f1'], amplitude=parameters['a1'])
        s2 = Sine(parameters['f2'], amplitude=parameters['a2'])
        return MixingFilter(s1,s2)

class InverSynthGenerator(SynthplayerGenerator):
    """
    Current best attempt to recreate the generator in the paper
    """
    def construct_generator(self,sample_rate:int,parameters:dict,extra:dict):
        """
        Paper
        #f,v,A,B are the carrier frequency, modulation frequency,
        # carrier amplitude and modulation amplitude, respectively

        each oscillator is associated with its own set of f,v,A,B
        All oscillators are frequency modulated by a sinusoidal waveform

        W = {sin,saw,tri,sqr}
        """
        p = parameters
        lfo1 = Sine(p['v1'], amplitude=p['B1'], samplerate=sample_rate)

        s1 = Sine(p['f1'], amplitude=p['A1'], fm_lfo=lfo1)

        lfo2 = Sine(p['v2'], amplitude=p['B2'], samplerate=sample_rate)
        s2 = Sawtooth(p['f2'], amplitude=p['A2'], fm_lfo=lfo2)

        lfo3 = Sine(p['v3'], amplitude=p['B3'], samplerate=sample_rate)
        s3 = Triangle(p['f3'], amplitude=p['A3'], fm_lfo=lfo3)

        lfo4 = Sine(p['v4'], amplitude=p['B4'], samplerate=sample_rate)
        s4 = Square(p['f4'], amplitude=p['A4'], fm_lfo=lfo4)

        y_osc = MixingFilter(s1,s2,s3,s4)

        """
        a (Attack) is the time taken for initial run-up of level from zero to peak,
        beginning when the key is first pressed.
        d (Decay) is the time taken for the subsequent run down from the attack
        level to the designated sustain level.
        s (Sustain) is the level during the main sequence of sound’s duration,
        until the key is released.
        r (Release) is the time taken for the level to decay from the sustain
        level to zero after the key is released.
        """
        y_env = EnvelopeFilter(y_osc,
            attack=p['attack'],
            decay=p['decay'],
            sustain=p['sustain_time'],
            sustain_level=p['sustain'],
            release=p['release'])

        """
        filter y_lp(𝑥, f_cut, q) that consists of a low-pass
        filter together with a resonance
        """
        y_lp = y_env

        """
        The gater is a Low Frequency Oscillator (LFO) that performs amplitude
        modulation to the input, according to a sine waveform with a frequency f_gate
        """
        y_gate = y_lp

        return y_gate



if __name__ == "__main__":
    import json
    gen = InverSynthGenerator()
    """
    * amplitudes are in [0.001, 1]
    * ADSR envelope parameters a, d, s, r are in [0.001, 1]
    * modulation amplitudes B_w are in [0, 1500] (where B_w = 0 means no frequency modulation)
    (not sure about this - 1500 seems large! 1.5?)
    * modulation frequency v_w in [1, 30]
    * gating frequency f_gate in [0.5, 30]
    * cutoff frequency f_cut in [200, 4000]
    * resonance q in [0.01, 10]
    """
    search_parameters = [
        Parameter("f1",freq_range(16)),
        Parameter("v1",param_range(16,1,30)),
        Parameter("A1",param_range(16,0., 1.)),
        Parameter("B1",param_range(16,0., 1.5)),

        Parameter("f2",freq_range(16)),
        Parameter("v2",param_range(16,1,30)),
        Parameter("A2",param_range(16,0., 1.)),
        Parameter("B2",param_range(16,0., 1.5)),

        Parameter("f3",freq_range(16)),
        Parameter("v3",param_range(16,1,30)),
        Parameter("A3",param_range(16,0., 1.)),
        Parameter("B3",param_range(16,0., 1.5)),

        Parameter("f4",freq_range(16)),
        Parameter("v4",param_range(16,1,30)),
        Parameter("A4",param_range(16,0., 1.)),
        Parameter("B4",param_range(16,0., 1.5)),

        #Parameter("attack",[0., 0.1, 0.2]),
        #Parameter("decay",[0., 0.1, 0.2]),
        #Parameter("sustain_time",[0.5, 0.7, 1]),
        #Parameter("sustain",[0., 0.5, 1.0]),
        #Parameter("release",[0., 0.1, 0.2]),
        #Parameter("release",[0., 0.1, 0.2]),
    ]
    # Sensible defaults for everything...
    fixed_parameters = {
        "f1":100,
        "v1":1,
        "A1":1.0,
        "B1":0.0,

        "f2":100,
        "v2":1,
        "A2":0.0,
        "B2":0.0,

        "f3":100,
        "v3":1,
        "A3":0.0,
        "B3":0.0,

        "f4":100,
        "v4":1,
        "A4":0.0,
        "B4":0.0,

        "attack":0.1,
        "decay":0.3,
        "sustain_time":0.5,
        "sustain":1.,
        "release":0.2,
    }
    print("-"*30)
    print("Search parameters:")
    for p in search_parameters:
        print("-- {}: {}".format(p.name,p.levels))
    print("-"*30)
    parameters=ParameterSet(
        parameters = search_parameters,
        fixed_parameters = fixed_parameters
    )
    g = DatasetCreator("inversynth_full",
        dataset_dir="test_datasets",
        wave_file_dir="test_waves/inversynth_full/",
        parameters=parameters
    )
    g.generate(sound_generator=gen,length=1,sample_rate=16384,method="random",max=40000)
