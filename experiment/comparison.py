from typing import List
from shutil import copyfile
from generators.vst_generator import VSTGenerator
from generators.parameters import load_parameter_set, ParameterSet
from models.app import load_model
from models.common.utils import utils
import re
from os import makedirs
import numpy as np


class Regenerator():
    def __init__(self,name,model,generator,parameters:ParameterSet,length=1.0,sample_rate=16384):
        self.name = name
        self.model = model
        self.generator = generator
        self.parameter_set = parameters
        self.sample_rate = sample_rate
        self.extra = {}
        self.extra['config'] = parameters
        self.length = 1.0

    def regenerate(self,input_filename:str,output_filename:str):
        # Get parameters from the model
        data = utils.wav_to_keras(input_filename,self.sample_rate)
        xdat = np.expand_dims(np.vstack([data]), axis=2)
        model_output = self.model.predict(xdat)[0]
        print(f"Model Output: {model_output}")
        settings = self.parameter_set.encoding_to_settings(model_output)

        print(f"Settings: {settings}")

        # Send parameters to generator
        self.generator.generate( parameters=settings,filename=output_filename,
                    length=self.length,sample_rate=self.sample_rate,extra=self.extra)

class Comparator():
    def __init__(self,models:List[Regenerator],output_dir="./comparison/by_model/"):
        self.models = models
        self.output_dir = output_dir

    def run_files(self,files:List[str]):
        for file in files:
            self.create_comparison(file)

    def create_comparison(self,input_filename):
        id = input_filename.replace(".wav","")
        id = re.sub(".*/","",id)
        output_dir = f"{self.output_dir}/{id}"
        makedirs(output_dir,exist_ok=True)
        copyfile(input_filename,self.get_filename(id,'original'))
        for model in self.models:
            fn = self.get_filename(id,model.name)
            model.regenerate(input_filename,fn)

    def get_filename(self,id,model_name):
        return f"{self.output_dir}/{id}/{id}_{model_name}.wav"

def spec_to_regenerator(spec:str):
    type,plugin_name,model_name,param_name = spec.split(":")

    vst = f"/Library/Audio/Plug-Ins/VST/{plugin_name}.vst"
    generator = VSTGenerator(vst=vst,sample_rate=44100)
    model = load_model(f"./output/{model_name}.h5")
    parameters = load_parameter_set(f"./plugin_config/{param_name}.json")
    return Regenerator(model_name,model,generator,parameters)

if __name__ == "__main__":
    import argparse
    print('Running with model...')
    parser = argparse.ArgumentParser(description='Run a model comparison')
    parser.add_argument('--model', type=str, dest='model', required=True, action='append',
                        help="""A list of model specifiers. Each one should have:
                        - a type (currently just 'vst')
                        - a name (for the VST plugin)
                        - a model name (e.g. dx2_fine_med)
                        - a paramter file name (e.g. dx2_fine)
                        all separated by colons: type:name:model:params,
                         e.g. vst:Dexed:dx2_fine_med_C6:dx2_fine""")
    parser.add_argument("--file", type=str,action='append')
    args = parser.parse_args()
    print(args.model)
    regenerators = [spec_to_regenerator(r) for r in args.model]
    comparator = Comparator(regenerators)
    comparator.run_files(args.file)
