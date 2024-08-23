import os
from os import path
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.collections import LineCollection
import pathlib
from matplotlib.lines import Line2D

DATASET_PATH = 'dataset'
nFFT = 512
RATE = 8000
BACK_COLOR = '#f7f7f7'
FRAGMENT_LENGTH = int(RATE * 0.4)
DURATION = round(1 / (RATE / FRAGMENT_LENGTH), 2)

basepath = path.dirname(__file__)
model_dir = pathlib.Path(os.path.join(basepath, '..', DATASET_PATH, 'model_{}s'.format(DURATION)))
model = tf.saved_model.load(model_dir)

def plot_remove_spines(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)


def to_chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

def get_chank_label_by_model(wave):
    x = tf.convert_to_tensor(wave, dtype=tf.float32)
    waveform =  x[tf.newaxis,...]
    result = None
    try:
      result = model(tf.constant(waveform))
    except Exception as e:
      return '-1'
    label_names = np.array(result['label_names'])
    prediction = tf.nn.softmax(result['predictions']).numpy()[0]
    max_value = max(prediction)
    i, = np.where(prediction == max_value)
    wave_label = label_names[i]
    return wave_label[0] if max_value > 0.7 else '-1'
 
x = os.path.join(basepath, '..', DATASET_PATH, 'valid', 'record.wav')
x = tf.io.read_file(str(x))
x, sample_rate = tf.audio.decode_wav(x, desired_channels=1, desired_samples=RATE * 2,)
x = tf.squeeze(x, axis=-1)
x = x[tf.newaxis,...]
waveform = x.numpy()[0];

chunks = to_chunks(waveform, int(FRAGMENT_LENGTH))

# Form segments for collection of lines   
segments = []
linecolors = []
colors = ['red', 'blue', '#67e77f', 'orange', 'yellow', 'brown', 'purple','black', 'black', 'black', 'grey']
   
x = 0
for lin_i, lin_y in enumerate(chunks):
  lineN = []
  lin_x = np.arange(len(lin_y))
  for i, y in enumerate(lin_y):
    # print('-',[x, y])
    lineN.append((x, y)) 
    x = x + 1
  segments.append(lineN)
  # windowed_lin_y= lin_y * np.hamming(len(lin_y))
  line_label = get_chank_label_by_model(lin_y)
  if('noise' in str(line_label)):
   color = colors[-1]
  else: 
    index = int(tf.strings.to_number(line_label)) or -1
    color = colors[index]
  linecolors.append(color)

# Create figure
fig, ax = plt.subplots(figsize=(12, 2))
fig.patch.set_facecolor(BACK_COLOR)
line_collection = LineCollection(segments=segments, colors=linecolors)
# Add a collection of lines
ax.add_collection(line_collection)

# Set x and y limits... sadly this is not done automatically for line
# collections
ax.set_xlim(0, len(waveform))
ax.set_ylim(1, -1)
legendColors = []
legendlabels = []
for i in np.arange(10):
   legendColors.append(Line2D([0, 1], [0, 1], color=colors[i]))
   legendlabels.append(str(i))
legendlabels[-1] = 'noise'   
ax.legend(legendColors, legendlabels)
ax.xaxis.set_major_locator(ticker.NullLocator())
ax.yaxis.set_major_locator(ticker.NullLocator())
ax.set_facecolor(BACK_COLOR)
plot_remove_spines(ax)
plt.show()