import numpy as np
import matplotlib.pyplot as plt

seq_id = "gMH_sFM_cAll_d24_mMH5_ch20"
music_file = "data/aistpp_20hz/"+seq_id+".mp3"
ddc_file = "data/aistpp_ddcpca/"+seq_id+".ddcpca.npy"
ddc_features = np.load(ddc_file)
import IPython.display as ipd
ipd.Audio(music_file) # load a local WAV file

import feature_extraction.madmom as madmom
from feature_extraction.madmom.audio.cepstrogram import MFCC
proc_dwn = madmom.features.RNNDownBeatProcessor()
beats = proc_dwn(music_file, fps=20)

%matplotlib inline
plt.matshow(beats[:200].T)

ddc_features.shape
beats.shape

plt.matshow(ddc_features[:200].T)

%matplotlib
plt.plot(ddc_features[:100,0])
plt.plot(beats[:100,0])


tgt_fps = 20
filtbank = madmom.audio.filters.MelFilterbank
spec = madmom.audio.spectrogram.Spectrogram(music_file, fps=tgt_fps, filterbank=filtbank, num_channels = 1)
# mfccs = MFCC(spec, filterbank=filtbank, num_bands=5)
# chroma = madmom.audio.chroma.PitchClassProfile(spec, num_classes=6, num_channels=1)
sectralflux = madmom.features.onsets.spectral_flux(spec)

mfccs.shape
sectralflux.shape
