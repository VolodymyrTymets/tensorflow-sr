# Tensorflow - Sound recognition Example

### Install dependencies

- Create and activate Python venv:

```shell
python3 -m venv localenv
source localenv/bin/activate
```

- Install dependencies:

`pip install -r requirements.txt`

### Train model

Create folder `dataset`. Download dataset from https://www.tensorflow.org/datasets/catalog/speech_commands. 
Place each digit record in own folder in `dataset/train` like `dataset/train/0`, `dataset/train/1` ... `dataset/train/9`

- train model:

`python3  train.py`

### Valid model

Create folder `dataset/valid`. 
Place digit record for validation in  folder in `dataset/valid`.

- valid model:

`python3  valid.py`

### Utils

- generate audio features graph:

`python3  audio_features.py`

- generate spectogram graph:

`python3  wave_to_spectogram.py`

- generate sample validation graph:

`python3  valid_on_samples.py`

- generate record validation graph:

`python3  valid_on_record.py`