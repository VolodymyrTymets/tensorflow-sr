import os
from os import listdir
from os.path import isfile, join
import pathlib
from os import path

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import tensorflow as tf

nFFT = 512
RATE = 8000
LINE_COLOR = '#67e77f'
BACK_COLOR = '#f7f7f7'
FRAGMENT_LENGTH = int(RATE  * 0.4)
DURATION = round(1 / (RATE / FRAGMENT_LENGTH), 2)

def plot_remove_spines(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)

def get_files(dir_path): 
  return [f for f in listdir(dir_path) if isfile(join(dir_path, f)) and f != '.DS_Store']

def get_wave(file_full_path):
  x = tf.io.read_file(str(file_full_path))
  x, sample_rate = tf.audio.decode_wav(x, desired_channels=1, desired_samples=FRAGMENT_LENGTH,)
  x = tf.squeeze(x, axis=-1)
  return x[tf.newaxis,...]

basepath = path.dirname(__file__)
valid_dir_path = os.path.join(basepath, '..', 'dataset', 'valid')
model_dir = pathlib.Path(os.path.join(basepath, '..', 'dataset' , 'model_{}s'.format(DURATION)))


files = get_files(valid_dir_path)
files.sort()
waves = []
names = []

model = tf.saved_model.load(model_dir)

for file in files:
  file_full_path = os.path.join(valid_dir_path, file)
  waveform = get_wave(file_full_path)
  waves.append(waveform);
  names.append(file)


## Plot
rows = len(waves)
cols = 2

fig, axes = plt.subplots(rows, cols, figsize=(6, 4), gridspec_kw={'width_ratios': [3, 2], 'hspace': 0.5})
fig.patch.set_facecolor(BACK_COLOR)
to_perc = np.vectorize(lambda x: x * 100)

for r in range(rows):
    w_ax, p_ax = axes[r]

    wave = waves[r]
    file_name = names[r]
    print('prediction for file:', file_name)
    
    result = model(tf.constant(wave))
    prediction = result['predictions']
    print('prediction:', tf.nn.softmax(prediction[0]))
    # wave
    ax = wave.numpy()[0]
    w_ax.plot(ax, color=LINE_COLOR)
    w_ax.set_ylim([-1.1, 1.1])
    w_ax.set_ylabel(file_name, fontweight ='bold')
    w_ax.xaxis.set_major_locator(ticker.NullLocator())
    w_ax.yaxis.set_major_locator(ticker.NullLocator())
    w_ax.set_facecolor(BACK_COLOR)
    plot_remove_spines(w_ax)

    # Prediction
    label_names = np.array(result['label_names'])
    yticks = to_perc(tf.nn.softmax(prediction[0]))
    p_ax.bar(label_names, yticks, color=LINE_COLOR)
    p_ax.set_facecolor(BACK_COLOR)
    plot_remove_spines(p_ax)

plt.show()
