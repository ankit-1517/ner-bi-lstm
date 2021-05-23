from ner_train_glove_random import *
from ner_train_char import *
import sys
import contextlib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from seqeval.metrics import classification_report as class_report
import copy
import argparse

# train_ner.py --initialization [random | glove ] --char_embeddings [ 0 | 1 ]
# --layer_normalization [ 0 | 1 ] --crf [ 0 | 1 ] --output_file <path to the
# trained model> --data_dir <directory containing data> --glove_embeddings_file
# <path to file containing glove embeddings> --vocabulary_output_file <path to
# the file in which vocabulary will be written>
parser = argparse.ArgumentParser()
parser.add_argument("--initialization", dest = "initialization")
parser.add_argument("--char_embeddings", dest = "char_embeddings", type=int)
parser.add_argument("--layer_normalization",dest ="layer_normalization")
parser.add_argument("--crf",dest ="crf")
parser.add_argument("--output_file",dest = "output_file")
parser.add_argument("--data_dir",dest = "data_dir")
parser.add_argument("--glove_embeddings_file", dest ="glove_embeddings_file")
parser.add_argument("--vocabulary_output_file", dest = "vocabulary_output_file")
args = parser.parse_args()

with_glove_embeddings = (args.initialization == "glove")
char_level_embeddings = args.char_embeddings
model_output_file = args.output_file
data_dir = args.data_dir
glove_embeddings_file = args.glove_embeddings_file
vocabulary_output_file = args.vocabulary_output_file

# specify GPU/CPU
if torch.cuda.is_available():
	device = torch.device('cuda')
else:
	device = torch.device('cpu')

if char_level_embeddings == 1:
    run_full_code_char(device, model_output_file, data_dir, glove_embeddings_file, vocabulary_output_file)
else:
    run_full_code_glove_random(device, with_glove_embeddings, model_output_file, data_dir, glove_embeddings_file, vocabulary_output_file)

