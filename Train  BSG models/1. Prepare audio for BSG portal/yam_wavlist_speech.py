# Copyright 2019 The TensorFlow Authors All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Inference demo for YAMNet."""
from __future__ import division, print_function

import sys

import numpy as np
import resampy
import soundfile as sf
import tensorflow as tf
import os

import params as yamnet_params
import yamnet as yamnet_model

# use only CPU
os.environ['CUDA_VISIBLE_DEVICES'] = ''

def main(argv):
  assert argv, 'Usage: yam_wavlist_max.py minprob listfile'

  params = yamnet_params.Params()
  yamnet = yamnet_model.yamnet_frames_model(params)
  yamnet.load_weights('yamnet.h5')
  yamnet_classes = yamnet_model.class_names('yamnet_class_map.csv')

  listfile=argv[0]
  
  fin = open(listfile,"r")

  spi = np.argmax(yamnet_classes == 'Speech')
  
  for line in fin:
    file_name = line.strip()
    # Decode the WAV file.
    wav_data, sr = sf.read(file_name, dtype=np.int16)
    assert wav_data.dtype == np.int16, 'Bad sample type: %r' % wav_data.dtype
    waveform = wav_data / 32768.0  # Convert to [-1.0, +1.0]
    waveform = waveform.astype('float32')

    # Convert to mono and the sample rate expected by YAMNet.
    if len(waveform.shape) > 1:
      waveform = np.mean(waveform, axis=1)
    if sr != params.sample_rate:
      waveform = resampy.resample(waveform, sr, params.sample_rate)

    # Predict YAMNet classes.
    scores, embeddings, spectrogram = yamnet(waveform)

    spmax = np.max(scores[:,spi])
    spmean = np.mean(scores[:,spi])
    spnum75 = np.sum(scores[:,spi] > 0.75)
    spnum50 = np.sum(scores[:,spi] > 0.50)
    spnum25 = np.sum(scores[:,spi] > 0.25)

    print(file_name + f'\t{spmax:.3f}\t{spmean:.3f}\t{spnum75}\t{spnum50}\t{spnum25}')
  fin.close()   

if __name__ == '__main__':
  main(sys.argv[1:])


