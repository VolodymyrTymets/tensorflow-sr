import numpy as np
from matplotlib import pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from definitions import BACK_COLOR, RATE


colors = ['red', 'orange', 'blue', '#67e77f', 'yellow', 'brown', 'purple','black', 'black', 'black', 'grey']

class Visualizer:
  def __init__(self, **kwargs):
    # # Create figure
    fig, ax = plt.subplots(figsize=(12, 2))
    fig.patch.set_facecolor(BACK_COLOR)
    self.fig = fig
    self.ax = ax

  def get_label_index(self, labels, line_label):
    try:
      return labels.tolist().index(line_label)
    except Exception as e:
      return -1
    
  def build_legend(self, labels):
    legendColors = []
    legendlabels = []

    for i in np.arange(len(labels)):
      label = labels[i];
      color = colors[self.get_label_index(labels=labels, line_label=label)]
      legendColors.append(Line2D([0, 1], [0, 1], color=color))
      legendlabels.append(str(label))

    label = 'unknown'
    color = colors[self.get_label_index(labels=labels, line_label=label)]
    legendColors.append(Line2D([0, 1], [0, 1], color=color))
    legendlabels.append(str(label))

    self.ax.legend(legendColors, legendlabels)
    
  def plot_remove_spines(self):
    self.ax.spines['top'].set_visible(False)
    self.ax.spines['right'].set_visible(False)
    self.ax.spines['bottom'].set_visible(False)
    self.ax.spines['left'].set_visible(False)  

  def build_axvline(self, segments, linecolors, timestamps):
    for i, segment in enumerate(segments):
      is_current = i > 0 and linecolors[i] == 'red'
      is_previous = i > 0 and linecolors[i-1] == 'red'
      is_next = i + 1 < len(segments) and linecolors[i+1] == 'red'
      if(is_current and is_next is False):
        self.ax.axvline(x=segment[-1][0], lw=1, ls='dashed', ymax=0.9, label='axvline - % of full height',)
        plt.text(x=segment[-1][0], y=-0.8, s=round(timestamps[i + 1 if i+1<len(segments) else i], 2))

      if(is_current and is_previous is False):
        self.ax.axvline(x=segment[0][0], lw=1, ls='dashed', ymax=0.9, label='axvline - % of full height')
        plt.text(x=segment[0][0], y=0.8, s=round(timestamps[i], 2))

  def show(self, segments, segment_labels, labels, timestamps, max_x):
    linecolors = [colors[self.get_label_index(labels=labels, line_label=l)] for l in segment_labels]
    line_collection = LineCollection(segments=segments, colors=linecolors)
    
    # Add a collection of lines
    self.ax.add_collection(line_collection)

    # Set x and y limits... sadly this is not done automatically for line
    # collections
    self.ax.set_xlim(0, max_x)
    self.ax.set_ylim(-1, 1)
    self.ax.xaxis.set_major_locator(ticker.MultipleLocator(RATE))
    self.ax.set_xticklabels([l.get_position()[0] / RATE for l in self.ax.get_xticklabels()])
    # self.ax.yaxis.set_major_locator(ticker.NullLocator())
    self.ax.set_facecolor(BACK_COLOR)
    self.plot_remove_spines()
    
    self.build_legend(labels=labels)
    self.build_axvline(segments=segments, linecolors=linecolors, timestamps=timestamps)
    plt.show()
