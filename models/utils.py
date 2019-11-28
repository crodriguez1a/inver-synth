import os
import re
import datetime
import librosa
import numpy as np

class Utils:
    @staticmethod
    def load_audio(path: str, sr: int, duration: float) -> tuple:
        y_audio, sample_rate = librosa.load(path,
                                            sr=sr, # `None` preserves sample rate
                                            duration=duration,)
        return (y_audio, sample_rate)

    @staticmethod
    def stft_to_audio(S: np.ndarray) -> np.ndarray:
        # Inverse STFT to audio
        return librosa.griffinlim(S)

    @property
    def fingerprint(self) -> str:
        timestamp: str = datetime.datetime.today().__str__()
        return re.sub(r'\.|\s|\\|\:', '', timestamp)

    def h5_save(self, model, save_path: str, filename_attrs: str='noattrs'):
        try:
            # save weights
            weights_f: str = f'{os.getcwd()}/{save_path}/model_weights_{filename_attrs}_{self.fingerprint}.h5'
            model.save_weights(weights_f)

            # write architecture to JSON
            arch_f: str = f'{os.getcwd()}/{save_path}/model_architecture_{filename_attrs}_{self.fingerprint}.json'
            with open(arch_f, 'w+') as f:
                f.write(model.to_json())

        except Exception as e:
            print(f"There was a problem saving the model: {e}")

# export initialized
utils: Utils = Utils()
