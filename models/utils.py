import librosa

class Utils:
    @staticmethod
    def load_audio(path: str, sr: int, duration: float) -> tuple:
        y_audio, sample_rate = librosa.load(path,
                                            sr=sr, # `None` preserves sample rate
                                            duration=duration,)
        return (y_audio, sample_rate)


# export initialized
utils: Utils = Utils()
