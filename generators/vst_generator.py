import json
import re
import sys

import librenderman as rm
import numpy as np
import samplerate
from scipy import stats

from generators.generator import *
from generators.parameters import *

sys.path.append(
    "/Users/dmrust//Uni/Work/Creative Computing/SynthLearn/RenderMan/Builds/MacOSX/build/Debug/"
)


class VSTGenerator(SoundGenerator):
    def __init__(
        self,
        vst: str,
        sample_rate,
        randomise_non_set: bool = True,
        randomise_all: bool = False,
    ):
        self.vst = vst
        self.randomise_non_set = randomise_non_set
        self.randomise_all = randomise_all
        self.sample_rate = sample_rate
        self.load_engine()

    def load_engine(self):
        try:
            print("_____ LOADING VST _______")
            engine = rm.RenderEngine(self.sample_rate, 512, 512)
            if engine.load_plugin(self.vst):
                print("Loaded {}".format(self.vst))

                self.engine = engine
                self.patch_generator = rm.PatchGenerator(engine)
            else:
                print("Couldn't load VST {}".format(self.vst))
            print("_____ LOADED VST _______")
        except Exception as e:
            print("Problem: {}".format(e))

    # def do_sound_generation(self,parameter_set,base_filename)->np.ndarray:
    def do_generate(
        self,
        parameters: dict,
        filename: str,
        length: float,
        sample_rate: int,
        extra: dict = {},
    ) -> np.ndarray:
        if not self.engine:
            print("VST not loaded")
            return np.zeros(5)
        resample = False
        if not self.sample_rate == sample_rate:
            resample = True

        engine = self.engine
        # print( engine.get_plugin_parameters_description() )
        # print("Params to set:{}".format(parameters))

        ids = dict([(p["name"], p["id"]) for p in extra["config"]["fixed_parameters"]])
        ids.update(dict([(p["name"], p["id"]) for p in extra["config"]["parameters"]]))

        # if self.randomise_non_set:
        # new_patch = self.patch_generator.get_random_patch()
        # engine.set_patch(new_patch)

        synth_params = dict(engine.get_patch())
        # Start with defaults

        # if not self.randomise_non_set:
        # for i in range(155):
        # synth_params[i] = 0.5

        for name, value in parameters.items():
            synth_params[ids[name]] = value

        if self.randomise_all:
            new_patch = self.patch_generator.get_random_patch()
            engine.set_patch(new_patch)

        note_length = length * 0.8
        if "note_length" in extra:
            note_length = extra["note_length"]

        engine.set_patch(list(synth_params.items()))
        engine.render_patch(40, 127, note_length, length)
        data = engine.get_audio_frames()

        if resample:
            ratio = sample_rate / self.sample_rate
            resamp = samplerate.resample(data, ratio, "sinc_best")
            # print(f"Resampling from {self.sample_rate} to {sample_rate}, ratio: {ratio}. Had {len(data)} samples, now {len(resamp)}")
            data = resamp

        nsamps_target = int(length * sample_rate)
        # print(f"Got {len(data)} frames as type {type(data)}. Target: {nsamps_target}")

        result = np.array(data[:nsamps_target])

        nans = np.sum(np.isnan(result))
        if nans > 0:
            print(f"Error: got {nans} NANs in file {filename}")
            return np.zeros(nsamps_target)
        return result

    def create_config(self, filename=None, default_value=0.0):
        r = re.compile("(.*): (.*)")
        params = []
        for line in self.engine.get_plugin_parameters_description().splitlines():
            m = r.match(line)
            if m:
                params.append(
                    {"id": int(m.group(1)), "name": m.group(2), "value": default_value}
                )
        output = {"parameters": [], "fixed_parameters": params}
        if filename:
            with open(filename, "w") as f:
                json.dump(output, f, indent=4)
        return output


# Run the generator to create a full dataset
def run_generator(args):  # name: str, plugin: str, config: str, max: int,
    # dataset_directory: str, wavefile_directory: str,
    # sample_rate: int = 16384, length: float = 1.0, note_length: float = -1, method: str = 'random'):

    note_length = args.note_length
    if note_length < 0.0:
        note_length = length * 0.8

    with open(args.config_file, "r") as f:
        config = json.load(f)
    sample = [
        Parameter(p["name"], p["values"], p.get("id", "")) for p in config["parameters"]
    ]
    fixed = dict([(p["name"], p["value"]) for p in config["fixed_parameters"]])

    plugin_rate = args.generate_samplerate or args.sample_rate

    generate_examples(
        gen=VSTGenerator(vst=args.plugin, sample_rate=plugin_rate),
        parameters=ParameterSet(parameters=sample, fixed_parameters=fixed),
        args=args,
        extra={"note_length": note_length, "config": config},
    )


# Create blank config file based on the plugin's parameter sets
def generate_defaults(plugin: str, output: str, default: float = 0.5):
    gen = VSTGenerator(vst=plugin, sample_rate=16384)
    gen.create_config(output, default_value=default)


# Example: python -m generators.vst_generator run --plugin /Library/Audio/Plug-Ins/VST/Lokomotiv.vst --config plugin_config/lokomotiv.json --dataset_name explore --wavefile_directory "test_waves/explore"

if __name__ == "__main__":
    pass
    # parser = argparse.ArgumentParser(description='Process some integers.')
    parser = default_generator_argparse()
    parser.add_argument(
        "command",
        type=str,
        choices=["run", "generate"],
        help="action to take: run (run the generator with a config) or generate (generate a blank config file for the plugin)",
    )
    parser.add_argument(
        "--plugin",
        dest="plugin",
        help='plugin file. .so on linux, on mac its the top level plugin dir, e.g. "/Library/Audio/Plug-Ins/VST/Lokomotiv.vst"',
    )
    parser.add_argument(
        "--output", dest="outfile", help="Place to store the generated parameters file"
    )
    parser.add_argument("--config", dest="config_file", help="Config file to use")
    parser.add_argument(
        "--default_value",
        type=float,
        dest="default_param",
        action="store",
        default=0.5,
        help="Default setting for parameters when generating a blank config",
    )
    parser.add_argument(
        "--note_length",
        type=float,
        dest="note_length",
        default=0.8,
        help="Length of a note in seconds",
    )
    parser.add_argument(
        "--generation_sample_rate",
        type=int,
        default=None,
        dest="generate_samplerate",
        help="Sample rate for audio generation. Defaults to target samplerate, but some plugins (Dexed) have trouble running a our funny sample rates. Will be resampled to the target rate after generation",
    )

    args = parser.parse_args()
    print(args)
    if args.command == "run":
        run_generator(args)
        # args.name, args.plugin, args.config_file,
        #          args.samples, args.data_dir, args.wave_dir)

    if args.command == "generate":
        generate_defaults(args.plugin, args.outfile, args.default_param)
    quit()
    run_locomotiv()
