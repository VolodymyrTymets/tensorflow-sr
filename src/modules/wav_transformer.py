import os
from os import path
import pathlib
from definitions import ROOT_DIR, RATE
import tensorflow as tf

class WavTransformer:
  def __init__(self, **kwargs):
    self.rate = pathlib.Path(os.path.join(ROOT_DIR, 'assets', os.getenv('CURRENT_SLUG'), 'dataset'))
  
  def to_chunks(self, lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]
  
  def get_wave_from_file(self, path, duration):
    file = tf.io.read_file(str(path))
    wave, sample_rate = tf.audio.decode_wav(file, desired_channels=1, desired_samples=RATE * duration)
    x = tf.squeeze(wave, axis=-1)
    x = x[tf.newaxis,...]
    waveform = x.numpy()[0]
    return waveform
