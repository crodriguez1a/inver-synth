import os
import librosa
import numpy as np
import uuid


class Utils:
    @staticmethod
    def load_audio(path: str, sr: int, duration: float) -> tuple:
        y_audio, sample_rate = librosa.load(
            path,
            # `sr=None` preserves sample rate
            sr=sr,
            duration=duration,
        )
        return (y_audio, sample_rate)

    @staticmethod
    def stft_to_audio(S: np.ndarray) -> np.ndarray:
        # Inverse STFT to audio
        return librosa.griffinlim(S)

    @staticmethod
    def write_audio(path: str, audio: np.ndarray, sample_rate: int = 44100):
        if path and os.path.isfile(path):
            librosa.output.write_wav(path, audio, sample_rate)
            print(f"Written successfully to {path}")

    @property
    def fingerprint(self) -> str:
        return uuid.uuid4()

    def h5_save(self, model, save_path: str):
        if not save_path:
            return

        try:
            base_path: str = f"{os.getcwd()}/{save_path}"

            # save weights
            weights_f: str = f"{base_path}/model_weights_{self.fingerprint}.h5"
            model.save_weights(weights_f)

            # write architecture to JSON
            arch_f: str = f"{base_path}/model_arch_{self.fingerprint}.json"
            with open(arch_f, "w+") as f:
                f.write(model.to_json())

        except Exception as e:
            print(f"There was a problem saving the model: {e}")

    @staticmethod
    def dummy_labels(size) -> np.ndarray:
        labels = np.random.uniform(size=size)
        return labels


# export initialized
utils: Utils = Utils()
