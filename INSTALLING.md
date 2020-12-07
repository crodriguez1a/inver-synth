

# Basic installation

- Set up boost before making a virtualenv (e.g. `conda install -c conda-forge boost`)
- Install packages with the `requirements-minimal.txt` file

# Setting up LibRenderman
Assuming anaconda's version of python - if not, replace paths with something appropriate

- Open up the Xcode project
- Make sure the following are in the Header search paths:
  - `/Users/dmurrayrust/opt/anaconda3/include/python3.8`
  - `/Users/dmurrayrust/opt/anaconda3/include`
- Make sure the following are in the Library search paths:
  - `/Users/dmurrayrust/opt/anaconda3/lib/`
- Change the "Other linker flags" section to `-shared -undefined dynamic_lookup -lboost_python38`,
  avoid having the `-lpython38` in there
- Build the project
- Move the newly built .so from the Renderman project to the root of the inver-synth directory (or link it - works better with rebuilding)
- Try running python, and do `import librenderman as rm` - this should print out a status message rather than failing.
- Try running `python -c "import librenderman as rm; rm.RenderEngine(44100,512,512)"` should not print errors
- Try `python -m generators.vst_generator generate --name dexed_install --plugin /Library/Audio/Plug-Ins/VST3/Dexed.vst3 --output test.json` or something similar.
