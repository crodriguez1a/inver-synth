import librenderman as rm
from generators.generator import *
from generators.parameters import *

import numpy as np
import math
from scipy import stats
import re
import json

import sys
sys.path.append(
    "/Users/dmrust//Uni/Work/Creative Computing/SynthLearn/RenderMan/Builds/MacOSX/build/Debug/")


class VSTGenerator(SoundGenerator):
    def __init__(self, vst: str, sample_rate, randomise_non_set: bool = True, randomise_all: bool = False):
        self.vst = vst
        self.randomise_non_set = randomise_non_set
        self.randomise_all = randomise_all
        self.sample_rate = sample_rate
        try:
            print("_____ LOADING VST _______")
            engine = rm.RenderEngine(sample_rate, 512, 512)
            if engine.load_plugin(self.vst):
                print("Loaded {}".format(self.vst))

                self.engine = engine
                self.patch_generator = rm.PatchGenerator(engine)
            else:
                print("Couldn't load VST {}".format(self.vst))
            print("_____ LOADED VST _______")
        except Exception as e:
            print("Problem: {}".format(e))

    #def do_sound_generation(self,parameter_set,base_filename)->np.ndarray:
    def do_generate(self,parameters:dict,filename:str,length:float,sample_rate:int,extra:dict={})->np.ndarray:
        if not self.engine:
            print("VST not loaded")
            return np.zeros(5)
        if not self.sample_rate == sample_rate:
            print("Mismatched sample rate. got {}, asked for {}".format(
                self.sample_rate, sample_rate))

        engine = self.engine
        #print( engine.get_plugin_parameters_description() )
        #print("Params to set:{}".format(parameters))

        ids = dict([(p['name'], p['id'])
                    for p in extra['config']['fixed_parameters']])
        ids.update(dict([(p['name'], p['id'])
                         for p in extra['config']['parameters']]))

        # if self.randomise_non_set:
        #new_patch = self.patch_generator.get_random_patch()
        # engine.set_patch(new_patch)

        synth_params = dict(engine.get_patch())
        # Start with defaults

        # if not self.randomise_non_set:
        # for i in range(155):
        #synth_params[i] = 0.5

        for name, value in parameters.items():
            synth_params[ids[name]] = value

        if self.randomise_all:
            new_patch = self.patch_generator.get_random_patch()
            engine.set_patch(new_patch)

        note_length = length * 0.8
        if 'note_length' in extra:
            note_length = extra['note_length']

        engine.set_patch(list(synth_params.items()))
        engine.render_patch(40, 127, note_length, length)
        data = engine.get_audio_frames()
        #print("Got {} frames as type {}".format(len(data),type(data)))
        nsamps = int(length*sample_rate)
        result = np.array(data[:nsamps])
        return result

    def create_config(self, filename=None, default_value=0.):
        r = re.compile("(.*): (.*)")
        params = []
        for line in self.engine.get_plugin_parameters_description().splitlines():
            m = r.match(line)
            if m:
                params.append({
                    "id": int(m.group(1)),
                    "name": m.group(2),
                    "value": default_value
                })
        output = {
            "parameters": [],
            "fixed_parameters": params
        }
        if filename:
            with open(filename, 'w') as f:
                json.dump(output, f, indent=4)
        return output


# Run the generator to create a full dataset
def run_generator(name: str, plugin: str, config: str, max: int,
                  dataset_directory: str, wavefile_directory: str,
                  sample_rate: int = 16384, length: float = 1.0, note_length: float = -1, method: str = 'random'):

    print("+"*50)
    print("Running generator '{}' to create {} points\n VST: {}\n Config: {}\n Dataset dir: {}\n Wave dir: {}".format(
        name, max, plugin, config, dataset_directory, wavefile_directory))
    gen = VSTGenerator(vst=plugin, sample_rate=sample_rate)

    if note_length < 0.0:
        note_length = length * 0.8

    with open(config, 'r') as f:
        config = json.load(f)

    sample = [Parameter(p['name'],p['values'],p.get('id',"")) for p in config['parameters']]
    fixed = dict([(p['name'],p['value']) for p in config['fixed_parameters']])
    parameters=ParameterSet(
        parameters = sample,
        fixed_parameters = fixed
    )
    g = DatasetCreator(name,
                       dataset_dir=dataset_directory,
                       wave_file_dir=wavefile_directory,
                       parameters=parameters
                       )
    print("+"*50)
    print("Starting Generation")
    g.generate(sound_generator=gen, length=length, sample_rate=sample_rate,
               method=method, max=max, extra={'note_length': note_length, 'config': config})
    print("+"*50)


# Create blank config file based on the plugin's parameter sets
def generate_defaults(plugin: str, output: str, default: float = 0.5):
    gen = VSTGenerator(vst=plugin, sample_rate=16384)
    gen.create_config(output, default_value=default)


# Example: python -m generators.vst_generator run --plugin /Library/Audio/Plug-Ins/VST/Lokomotiv.vst --config plugin_config/lokomotiv.json --dataset_name explore --wavefile_directory "test_waves/explore"

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('command',  type=str, choices=['run', 'generate'],
                        help='action to take: run (run the generator with a config) or generate (generate a blank config file for the plugin)')
    parser.add_argument('--plugin', dest='plugin', action='store',
                        help='plugin file. .so on linux, on mac its the top level plugin dir, e.g. "/Library/Audio/Plug-Ins/VST/Lokomotiv.vst"')
    parser.add_argument('--output', dest='outfile', action='store',
                        help='Place to store the output file')
    parser.add_argument('--config', dest='config_file', action='store',
                        help='Config file to use')
    parser.add_argument('--num_examples', type=int, dest='samples', action='store', default=50,
                        help='Number of examples to create')
    parser.add_argument('--dataset_name', type=str, dest='name', action='store', default='unknown',
                        help='Name of datasets to create')
    parser.add_argument('--dataset_directory', type=str, dest='data_dir', action='store', default='test_datasets',
                        help='Directory to put datasets')
    parser.add_argument('--wavefile_directory', type=str, dest='wave_dir', action='store', default='test_waves/unknown/',
                        help='Directory to put wave files')
    parser.add_argument('--default_value', type=float, dest='default_param', action='store', default=0.5,
                        help='Default setting for parameters when generating a blank config')

    args = parser.parse_args()
    print(args)
    if args.command == 'run':
        run_generator(args.name, args.plugin, args.config_file,
                      args.samples, args.data_dir, args.wave_dir)
        pass
    if args.command == 'generate':
        generate_defaults(args.plugin, args.outfile, args.default_param)
        pass
    quit()
    run_locomotiv()
