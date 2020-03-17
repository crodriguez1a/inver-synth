
# Sinewave generator

A pure Python super simple test generator that just has two sinewaves added together
with control over amplitude and frequency of each.

# VST generators

We can wrap any VST plugin using the RenderMan engine (https://github.com/fedden/RenderMan)

Making this work is slightly complex just due to VST handling and the number of parameters.

## Installation

Install RenderMan - this is a bit tricky, may involve hacking XCode build files - don't have a good recipe yet!

## Setup Parameters
First, generate a default parameter set for your plugin using the `generate` command:
```
python -m generators.vst_generator generate --plugin /Library/Audio/Plug-Ins/VST/Lokomotiv.st --output test.json
```

This loads the Lokomotiv plugin from the default macos VST install location. The
first argument 'generate' means generate an empty config file, and the --output
gives a location to create it.

The file will look a bit like:
```
{
    "parameters": [],
    "fixed_parameters": [
        {
            "id": 0,
            "name": "BPM Sync On/Off",
            "value": 0.5
        },
```
with each paramter having an id (int) and a name, as well as a default value.

Now, edit this file to set out the search space. Fixed parameters will be passed
to the synth as is, while the general parameters are given a set of levels to
search over, e.g.:
```
"parameters": [

     {
         "id": 1,
         "name": "LFO Type select",
         "values": [0.0, 0.5, 1.0]
     },
     {
         "id": 2,
         "name": "LFO Speed",
         "values": [0.0, 0.5, 1.0]
     },
```
By moving things between fixed and open parameters, and changing the value sets,
you can expand or contract the search space, and get rid of parameters you don't
want to learn.

## Generate Dataset
Using the `run` command, generate the dataset:
```
python -m generators.vst_generator run --plugin /Library/Audio/Plug-Ins/VST/Lokomotiv.vst --config plugin_config/lokomotiv.json --dataset_name explore --wavefile_directory "test_waves/explore"
```
