import os
import numpy as np
from dotenv import load_dotenv
load_dotenv()
from src.modules.file_worker import file_worker
from src.modules.wav_transformer import WavTransformer
from src.modules.recognizer import Recognizer
from src.modules.visualizer import Visualizer
from definitions import RATE

FRAGMENT_DURATION = float(os.getenv('TRAINED_DURATION_IN_SECONDS'))
FRAGMENT_LENGTH = int(RATE * FRAGMENT_DURATION)

def main ():
  recognizer = Recognizer()
  wav_transformer = WavTransformer()
  visualizer = Visualizer()
  file_path = os.path.join(file_worker.get_data_set_path(),  'valid', 'long', os.getenv('CURRENT_FILE')+'.wav')
  waveform = wav_transformer.get_wave_from_file(path=file_path, desired_samples=RATE * 21)
  chunks = wav_transformer.to_chunks(waveform, FRAGMENT_LENGTH)
  _, labels = recognizer.get_chank_label_by_model(np.zeros(FRAGMENT_LENGTH))

  segments = []
  segment_labels = []
  timestamps = []
  x = 0

  # # Form segments for collection of lines  
  for lin_i, lin_y in enumerate(chunks):
    lineN = []
    for i, y in enumerate(lin_y):
      lineN.append((x, y)) 
      x = x + 1
    timestamps.append(lin_i * FRAGMENT_DURATION)
    segments.append(lineN)
    segment_label, _ = recognizer.get_chank_label_by_model(lin_y)
    segment_labels.append(segment_label)
    # windowed_lin_y= lin_y * np.hamming(len(lin_y))
    
  visualizer.show(segments=segments, segment_labels=segment_labels, labels=labels, timestamps=timestamps, max_x=len(waveform))


main();  