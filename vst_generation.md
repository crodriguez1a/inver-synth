# VST generation setup

librenderman is a bit fiddly. As a result, the .so file is currently in the root directory. This is not a good write up, but it's what mo-seph has done to make it work on a macbook.

It is compiled to work with a particular Python version, as it's tricky making it work with XCode builds.

This means changing the XCode build file:
- header search paths: `../../VST3_SDK ../../JuceLibraryCode ../../JuceLibraryCode/modules /usr/include/python /usr/local/include /usr/local/Cellar/python/3.7.7/Frameworks/Python.framework/Versions/3.7/include/python3.7m /usr/local/Cellar/python/3.7.7/Frameworks/Python.framework/Versions/3.7/include/python3.7m $(inherited)`
- Library search paths: `$(inherited) /usr/local/lib /usr/local/Cellar/python/3.7.7/Frameworks/Python.framework/Versions/3.7/lib/`
- Other linker flags: `-shared -lpython3.7m -lboost_python37`

This means it only works for the given version of Python, and it
doesn't seem happy to work in a Virtualenv.

So, to use librenderman, the current command is `/usr/local/Cellar/python/3.7.7/bin/python3.7m`

This also means that modules need to be installed for this version of python, by running `/usr/local/Cellar/python/3.7.7/bin/python3.7m -m pip install <module>` etc.
