"""
End-to-End learning. A CNN predicts the synthesizer
parameter configuration directly from the raw audio.
The first convolutional layers perform 1D convolutions
that learn an alternative representation for the STFT
Spectrogram. Then, a stack of 2D convolutional layers
analyze the learned representation to predict the
synthesizer parameter configuration.
"""
