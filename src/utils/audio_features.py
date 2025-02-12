import os
from os import path

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import tensorflow as tf

nFFT = 512
RATE = 8000
LINE_COLOR = '#67e77f'
BACK_COLOR = '#f7f7f7'

def plot_remove_spines(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)

def get_spectrogram(waveform):
  # Convert the waveform to a spectrogram via a STFT.
  spectrogram = tf.signal.stft(
      waveform, frame_length=256, frame_step=128)

  # Obtain the magnitude of the STFT.
  spectrogram = tf.abs(spectrogram)
  # Add a `channels` dimension, so that the spectrogram can be used
  # as image-like input data with convolution layers (which expect
  # shape (`batch_size`, `height`, `width`, `channels`).
  spectrogram = spectrogram[..., tf.newaxis]
  return spectrogram

def plot_spectrogram(spectrogram, ax):
  if len(spectrogram.shape) > 2:
    assert len(spectrogram.shape) == 3
    spectrogram = np.squeeze(spectrogram, axis=-1)
  # Convert the frequencies to log scale and transpose, so that the time is
  # represented on the x-axis (columns).
  # Add an epsilon to avoid taking a log of zero.
  log_spec = np.log(spectrogram.T + np.finfo(float).eps)
  height = log_spec.shape[0]
  width = log_spec.shape[1]
  X = np.linspace(0, np.size(spectrogram), num=width, dtype=int)
  Y = range(height)
  ax.pcolormesh(X, Y, log_spec)

basepath = path.dirname(__file__)
path = os.path.join(basepath, '..', 'dataset', 'train', '5', '5_george_45.wav')
x = tf.io.read_file(str(path))
x, sample_rate = tf.audio.decode_wav(x, desired_channels=1, desired_samples=RATE * 0.4,)
x = tf.squeeze(x, axis=-1)
x = x[tf.newaxis,...]
waveform = x.numpy()[0];

## Plot
rows = 1
cols = 3

fig, (w_ax, s_ax, sg_ax) = plt.subplots(rows, cols)
fig.patch.set_facecolor(BACK_COLOR)

# wave
w_ax.plot(waveform, color=LINE_COLOR)
w_ax.set_ylim([-1.1, 1.1])
w_ax.set_title('Time')
w_ax.xaxis.set_major_locator(ticker.NullLocator())
w_ax.yaxis.set_major_locator(ticker.NullLocator())
w_ax.set_facecolor(BACK_COLOR)
plot_remove_spines(w_ax)

# Spectr

MAX_y = 2.0 ** (2 * 8 - 1) * 2
y_L = waveform[::2]
y_R = waveform[1::2]
Y_L = np.fft.fft(y_L, nFFT)
Y_R = np.fft.fft(y_R, nFFT)
# Sewing FFT of two channels together, DC part uses right channel's
spectr = abs(np.hstack((Y_L[int(-nFFT / 2):-1], Y_R[:int(nFFT / 2)]))) / MAX_y

x_f = 1.0 * np.arange(-nFFT / 2 + 1, nFFT / 2) / nFFT * RATE

s_ax.set_yscale('linear')
s_ax.yaxis.set_major_locator(ticker.NullLocator())
s_ax.plot(spectr, color=LINE_COLOR)
s_ax.set_title('Frequency')
s_ax.xaxis.set_major_locator(ticker.NullLocator())
s_ax.yaxis.set_major_locator(ticker.NullLocator())
s_ax.set_facecolor(BACK_COLOR)
plot_remove_spines(s_ax)

# Spectogram
spectrogram = get_spectrogram(waveform)
plot_spectrogram(spectrogram.numpy(), sg_ax)
sg_ax.set_title('Time and Frequency')
sg_ax.xaxis.set_major_locator(ticker.NullLocator())
sg_ax.yaxis.set_major_locator(ticker.NullLocator())
plot_remove_spines(sg_ax)

plt.show()