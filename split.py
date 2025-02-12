import os
from os import listdir
from os.path import isfile, join
import wave
import struct
import uuid
import numpy as np
from os import path
from dotenv import load_dotenv
load_dotenv()
from definitions import RATE, nFFT
from src.modules.file_worker import file_worker


FRAGMENT_DURATION = float(os.getenv('TRAINED_DURATION_IN_SECONDS'))
FRAGMENT_LENGTH = int(RATE * FRAGMENT_DURATION)

class Fragmenter:
    def __init__(self, model, out_folder):
        self.sample_size = None
        self.fragment = []
        self.model = model;
        self.file_name = None
        self.out_folder = out_folder
        self.counter = 0
    

    def create_folder(self, directory: str):
        if not os.path.exists(directory):
            os.makedirs(directory)

    def save_fragment(self, chunk):
        self.fragment = np.concatenate((self.fragment, chunk))
   
    def clear_fragment(self):
        self.fragment = []
    
    def buffer_to_chunk(self, in_data, chanels_count): 
        y = []
        try:
            y = np.array(struct.unpack("%dh" % (chanels_count * nFFT), in_data))
        except:
            print("An exception occurred")
            return
        y_L = y[::2]
        y_R = y[1::2]
        chunk = np.hstack((y_L, y_R))
        return chunk
        
            
    def split(self, buffer, source_file, file_name):
        self.file_name = file_name
        chunk = self.buffer_to_chunk(in_data=buffer, chanels_count=source_file.getnchannels())
        if(chunk is None):
            return
        if(len(self.fragment) <  FRAGMENT_LENGTH):
            self.save_fragment(chunk);
        else:
            self.write_chunk(self.fragment, source_file)
            self.clear_fragment()
            self.save_fragment(chunk)


    def write_chunk(self, chunk, source_file):
        self.create_folder(self.out_folder)
        self.counter = self.counter + 1;
        file_name = os.path.join(self.out_folder, '{}_{}.wav'.format(self.counter, uuid.uuid4()))
        # print('--> write to:', file_name)
        wav_file = wave.open(file_name, 'w')
        wav_file.setparams(
            (1, source_file.getsampwidth(), source_file.getframerate(), source_file.getnframes(), "NONE", "not compressed"))
        for sample in chunk:
            wav_file.writeframes(struct.pack('h', int(sample)))


def get_only_files(path):
    return [f for f in listdir(path) if isfile(join(path, f)) and f != '.DS_Store']

def append_duration(name):
    return '{}_{}'.format(FRAGMENT_DURATION, name)

def split(path, out_path):
    print('Start split to {}ms for {}'.format(FRAGMENT_DURATION, path))
    mode = None;
    fragmenter = Fragmenter(mode, out_path)
    files = get_only_files(path)
    
    for file in files:
        file_path = os.path.join(path, file)
        print('--> read from:', file_path)
        wav_file = wave.open(file_path, 'rb')
        data = wav_file.readframes(nFFT)
        while data != b'':
            fragmenter.split(data, wav_file, file)
            data = wav_file.readframes(nFFT)

                 
basepath = path.dirname(__file__)
ASSETSS_FOLDER = 'dataset/train'
print('Split into Duration: {}'.format(FRAGMENT_DURATION))
split(os.path.join(file_worker.get_data_set_path(), 'train', 'spray'), os.path.join(file_worker.get_data_set_path(), 'train', 'spray', 'split'))