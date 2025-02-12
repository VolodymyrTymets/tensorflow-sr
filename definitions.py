import os
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
nFFT = 512
RATE = 8000
BACK_COLOR = '#f7f7f7'
FRAGMENT_LENGTH = int(RATE * 0.2)
EPOCHS = 10