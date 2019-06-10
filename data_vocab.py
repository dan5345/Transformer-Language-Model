import collections
import os
import torch
import json
from utils import pad_sents, read_corpus
import argparse

def get_args():
    parser = argparse.ArgumentParser('build vocab')
    parser.add_argument('--data-dir')
    return parser.parse_args()


class Dictionary(object):
    def __init__(self,pre_dict = None):
        if pre_dict:
            self.word2id = pre_dict
        else:
            self.word2id = dict()
            self.word2id['<pad>'] = 0
            self.word2id['<s>'] = 1
            self.word2id['</s>'] = 2
            self.word2id['<UNK>'] = 3
        self.id2word = {k:v for k,v in self.word2id.items()}
        self.unk_id = self.word2id['<UNK>']

    def __len__(self):
        return len(self.word2id)


    def __getitem__(self,word):
        return self.word2id.get(word,self.unk_id)

    def add_word(self,word):
        if word not in self.word2id:
            idx = len(self.word2id)
            self.word2id[word] = idx
            self.id2word[idx] = word
        idx = None
        return idx

    def words2indices(self,sents):
        if type(sents[0]) == list:
            return [[self.word2id[w] for w in s] for s in sents]
        else:
            return [self.word2id[w] for w in sents]

    def indices2words(self,word2ids):
        return [self.id2word[w_id] for w_id in word2ids]


    def to_input_tensor(self,sents,device):
        word_ids = self.words2indices(sents)
        sents_t = pad_sents(word_ids, self.word2id['<pad>'])
        sents_var = torch.tensor(sents_t, dtype = torch.long, device = device)
        return torch.t(sents_var)

    @staticmethod
    def from_corpus(corpus):
        vocab = Dictionary()
        for sents in corpus:
            for word in sents:
                vocab.add_word(word)

        return vocab


class Vocab(object):
    def __init__(self,vocab):
        self.vocab = vocab

    def __len__(self):
        return len(self.vocab)


    @staticmethod
    def build(sents):
        vocab = Dictionary.from_corpus(sents)
        return Vocab(vocab)

    def save(self, file_path):
        json.dump(dict(src_word2id = self.vocab.word2id), open(file_path, 'w'), indent = 2)

    @staticmethod
    def load(file_path):
        entry = json.load(open(file_path, 'r'))
        word2id = entry['src_word2id']
        return Dictionary(word2id) 

if __name__ == '__main__':
    args = get_args()
    sents = read_corpus(args.data_dir)
    vocab = Vocab.build(sents)
    vocab.save('vocab.json')
