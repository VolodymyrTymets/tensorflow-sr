import os
import pathlib
from src.modules.file_worker import file_worker
import tensorflow as tf
import numpy as np

class Recognizer:
  def __init__(self, **kwargs):
    self.file_worker = file_worker;

    model_dir = pathlib.Path(file_worker.get_assets_path(), 'models', 'model_{}s'.format(os.getenv('TRAINED_DURATION_IN_SECONDS')))
    self.model = tf.saved_model.load(model_dir)

  def get_chank_label_by_model(self, wave):
      x = tf.convert_to_tensor(wave, dtype=tf.float32)
      waveform =  x[tf.newaxis,...]
      result = None
      try:
        result = self.model(tf.constant(waveform))
      except Exception as e:
        return '-1'
      label_names = np.array(result['label_names'])
      prediction = tf.nn.softmax(result['predictions']).numpy()[0]
      max_value = max(prediction)
      i, = np.where(prediction == max_value)
      wave_label = label_names[i]
      return wave_label[0] if max_value > 0.7 else '-1', label_names