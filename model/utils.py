import torch
import math
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import argparse
import collections 
import sys

def get_args():
    parser = argparse.ArgumentParser("Preparing data")
    parser.add_argument("--data-dir", default = None, help='path to training data')
    parser.add_argument("--valid-dir", default = None, help = 'path to valid data')
    parser.add_argument("--test-dir", default = None, help = "path to test data")
    parser.add_argument("--mode", default = None, help = "type: str, {'train','test'}")
    parser.add_argument("--cuda", default = False, help = "use GPU")
    parser.add_argument("--seed", default = 2019 , help = "random seed")
    parser.add_argument("--batch-size", default = 128, help= "training batch size")
    parser.add_argument("--clip-grad", default = 5.0, help ="gradient clipping")
    parser.add_argument("--valid-freq", default = 1000,type = int, help="number of batches between each validation")
    parser.add_argument("--model-save-path", default = "model.bin", help="model save path")
    parser.add_argument("--embed-size", default = 256, help = "embedding size")
    parser.add_argument("--hidden-size", default = 256, help = "hidden size for rnn ")
    parser.add_argument("--dropout", default = 0.1, help ="dropout rate")
    parser.add_argument("--tie-embed", default = False, help = "using tie embedding")
    parser.add_argument("--save-path", default = "./saved_model/model.bin", help = "save directory")

    return parser.parse_args()





def pad_sents(sents,pad_token):
    sents_padded = []

    max_len = np.max([len(sent) for sent in sents])
    for sent in sents:
        while len(sent) < max_len:
            sent.append(pad_token)

        sents_padded.append(sent)

    return sents_padded


def read_corpus(file_path):
    data = []
    for line in open(file_path):
        sent = line.strip().split(' ')
        sent = ['<s>'] + sent + ['</s>']
        data.append(sent)

    return data


def batch_iter(data,batch_size, shuffle = False):
    batch_num = math.ceil(len(data) / batch_size)
    index_array = list(range(len(data)))

    if shuffle :
        np.random.shuffle(index_array)

    for i in range(batch_num):
        indices = index_array[i*batch_size:(i+1)*batch_size]
        examples = [data[idx] for idx in indices]
        examples = sorted(examples, key = lambda e : len(e), reverse = True)
        yield [e for e in examples]


