

# Setting up the runs

## Data directory
- Make a data directory somewhere that's happy to have big files, e.g. `mkdir ~/InverSynthData`
- Create a `waves` and a `datasets` directory in there, and link them in the main directory, e.g. `ln -s ~/InverSynthData/waves main_waves`

- Generate audio samples with a command like this: `python -m generators.vst_generator run --name dx6_fine_large --plugin /Library/Audio/Plug-Ins/VST/Dexed.vst --config plugin_config/dx6_fine.json --num_examples 40000 --generation_sample_rate 48000 --dataset_directory datasets --wavefile_directory waves`
  - run through (coarse|fine) and (dx2|dx6)
  - sizes: large = 40k examples , huge = ???


- Train the model:
  - `python -m models.e2e_cnn --model C6 --dataset_name dx2_coarse_large --dataset_dir datasets --parameters_file plugin_config/dx2_coarse.json`
  - 100 epochs is default
