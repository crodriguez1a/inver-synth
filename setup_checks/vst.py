import librenderman as rm
import faulthandler
faulthandler.enable()

engine = rm.RenderEngine(44100,512,512)
path = "/Library/Audio/Plug-Ins/VST/Dexed.vst"
if engine.load_plugin(path):
    print("Loaded OK")

generator = rm.PatchGenerator(engine)
random_patch = generator.get_random_patch()
engine.set_patch(random_patch)
engine.render_patch(40, 127, 4.0, 5.0)
