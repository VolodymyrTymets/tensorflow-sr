import os
from os import path

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import tensorflow as tf
import tensorflow_io as tfio

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

def get_waveform(path):
  x = tf.io.read_file(str(path))
  x, sample_rate = tf.audio.decode_wav(x, desired_channels=1, desired_samples=RATE * 0.4,)
  x = tf.squeeze(x, axis=-1)
  x = x[tf.newaxis,...]
  return x.numpy()[0];

basepath = path.dirname(__file__)

def get_data(path):
  waveform = get_waveform(path)
  position = tfio.audio.trim(waveform, axis=0, epsilon=0.1).numpy()
  trim_wave = waveform[position[0]:position[1]]
  spectrogram = get_spectrogram(trim_wave).numpy()
  return [waveform, spectrogram]

data_set = []
for digit in range(6):
  path = os.path.join(basepath, '..', 'dataset', 'train', str(digit), '{}_george_45.wav'.format(digit))
  data_set.append(get_data(path))

## Plot
rows = len(data_set)
cols = len(data_set[0])
# fig, (w_ax, hm_ax, sg_ax) = plt.subplots(rows, cols)

fig, axes = plt.subplots(rows, cols, figsize=(16, 9))
fig.patch.set_facecolor(BACK_COLOR)

for r in range(rows):
    w_ax, sg_ax = axes[r]
    wave,spectogram = data_set[r];
    # wave
    w_ax.plot(wave, color=LINE_COLOR)
    w_ax.set_facecolor(BACK_COLOR)
    w_ax.set_ylim([-1.1, 1.1])
    if(r == 0):
      w_ax.set_title('Time')
    w_ax.set_ylabel('{}.wav'.format(r + 1), fontweight ='bold')
    w_ax.xaxis.set_major_locator(ticker.NullLocator())
    w_ax.yaxis.set_major_locator(ticker.NullLocator())
    plot_remove_spines(w_ax)

    # Spectogram
    plot_spectrogram(spectogram, sg_ax)
    if(r == 0):
      sg_ax.set_title('Time and Frequency')
    sg_ax.xaxis.set_major_locator(ticker.NullLocator())
    sg_ax.yaxis.set_major_locator(ticker.NullLocator())
    plot_remove_spines(w_ax)

plt.show()