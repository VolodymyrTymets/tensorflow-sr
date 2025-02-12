import os
from os import path
import pathlib
from definitions import ROOT_DIR

class FileWorker:
  def __init__(self, **kwargs):
    self.assets_path = pathlib.Path(os.path.join(ROOT_DIR, 'assets', os.getenv('CURRENT_SLUG')))
    self.data_set_path = pathlib.Path(os.path.join(self.assets_path , 'dataset'))

  def get_assets_path(self,):
    return self.assets_path
  
  def get_data_set_path(self,):
    return self.data_set_path
  

file_worker = FileWorker()