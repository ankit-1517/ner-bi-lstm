from ner_test_glove_random import *
from ner_test_char import *
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

# python3 test_ner.py --initialization glove --model_file <path to the trained
# model> --char_embeddings [ 0 | 1 ] --layer_normalization [ 0 | 1 ] --crf [ 0 |
# 1 ] --test_data_file <path to a file in the same format as original train file
# with random  NER / POS tags for each token> --output_file <file in the same
# format as the test data file with random NER tags replaced with the
# predictions> --glove_embeddings_file <path to file containing glove
# embeddings> --vocabulary_input_file <path to the vocabulary file written while
# training>
parser = argparse.ArgumentParser()
parser.add_argument("--initialization", dest = "initialization")
parser.add_argument("--model_file", dest = "model_file")
parser.add_argument("--char_embeddings", dest = "char_embeddings", type=int)
parser.add_argument("--layer_normalization", dest = "layer_normalization", type=int)
parser.add_argument("--crf", dest = "crf", type=int)
parser.add_argument("--test_data_file",dest ="test_data_file")
parser.add_argument("--output_file",dest = "output_file")
parser.add_argument("--glove_embeddings_file", dest ="glove_embeddings_file")
parser.add_argument("--vocabulary_input_file", dest = "vocabulary_input_file")
args = parser.parse_args()

model_file = args.model_file
char_level_embeddings = args.char_embeddings
test_data_file = args.test_data_file
output_file = args.output_file
glove_embeddings_file = args.glove_embeddings_file
vocabulary_input_file = args.vocabulary_input_file

# specify GPU/CPU
if torch.cuda.is_available():
	device = torch.device('cuda')
else:
	device = torch.device('cpu')

if char_level_embeddings == 1:
    run_full_code_char(device, model_file, test_data_file, output_file, vocabulary_input_file)
else:
    run_full_code_glove_random(device, model_file, test_data_file, output_file, vocabulary_input_file)

