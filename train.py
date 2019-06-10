import math
import sys
import time
from lm import RNNLM
import numpy as np
from tqdm import tqdm
from data_vocab import Vocab
import logging

from utils import read_corpus, batch_iter, get_args
import torch
import torch.nn.utils

def evaluate_ppl(model, dev_data, batch_size = 32):
    """
    Calculate validation perplexity
    """
    was_training = model.training
    model.eval()

    cum_loss = 0.
    total_word = 0.
    with torch.no_grad():
        for sents in batch_iter(dev_data,batch_size):
            loss = -model(sents).sum()
            cum_loss += loss.item()
            target_word = sum(len(s[1:]) for s in sents)
            total_word +=target_word

        ppl = np.exp(cum_loss/total_word)

    if was_training:
        model.train()

    return ppl

def train(args):
    """
    Perform training
    """
    train_data = read_corpus(args.data_dir)
    dev_data = read_corpus(args.valid_dir)
    train_batch_size = args.batch_size
    clip_grad = args.clip_grad 
    valid_freq = args.valid_freq 
    save_path = args.model_save_path
    vocab = Vocab.load('vocab.json')
    device = torch.device("cuda:0" if args.cuda else "cpu")
    max_patience = 5
    max_num_trial  = 5
    learning_rate_decay = 0.5
    max_epoch = 5000

    model = RNNLM(embed_size = args.embed_size, hidden_size = args.hidden_size, vocab = vocab, dropout_rate = args.dropout, device = device, tie_embed = args.tie_embed)

    model.train()
    
    #Xavier initialization
    for p in model.parameters():
        p.data.uniform_(-0.1,0.1)

    model.to(device)

    #TODO Tunable learning rate
    optimizer = torch.optim.Adam(model.parameters(), lr = 0.001)

    num_trial = 0
    train_iter = patience = cum_loss = report_loss = total_word = report_total_word = 0

    cum_examples = report_examples = epoch = valid_num = 0

    hist_valid_scores = []
    train_time = begin_time = time.time()

    print("Begin training")

    while True:
        epoch +=1

        for sent_batch in batch_iter(train_data,batch_size = train_batch_size, shuffle = True):
            train_iter += 1

            optimizer.zero_grad()

            batch_size = len(sent_batch)

            example_losses = -model(sent_batch)

            batch_loss  = example_losses.sum()
            loss = batch_loss / batch_size
            
            loss.backward()

            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)

            optimizer.step()

            batch_losses_val = batch_loss.item()
            report_loss += batch_losses_val
            cum_loss += batch_losses_val

            tgt_word_num_to_predict = sum(len(s[1:]) for s in sent_batch)
            total_word += tgt_word_num_to_predict

            report_total_word += tgt_word_num_to_predict
            report_examples += batch_size
            cum_examples += batch_size

            if train_iter % 10 ==0:
                print('epoch %d, iter %d, avg.loss %.2f, avg. ppl %.2f' \
                    'cum. examples %d, speed %.2f words/sec, time elapsed %.2f sec' %(epoch, train_iter, report_loss/ report_examples, math.exp(report_loss / report_total_word),cum_examples, report_total_word / (time.time() - train_time), time.time() - begin_time), file = sys.stderr)

                train_time = time.time()
                report_loss = report_total_word = report_examples = 0.


            #VALIDATION
            if train_iter % valid_freq == 0 :
                print("epoch %d, iter %d, cum. loss %.2f, cum. ppl %.2f, cum. examples %d" % (epoch, train_iter, cum_loss/ cum_examples, np.exp(cum_loss/ total_word), cum_examples), file = sys.stderr)

                cum_loss = cum_examples = total_word = 0

                valid_num +=1

                print("Begin validation", file = sys.stderr)

                dev_ppl = evaluate_ppl(model, dev_data,batch_size = 128)
                valid_metric = -dev_ppl

                print("validation: iter %d, dev. ppl %f" % (train_iter,dev_ppl), file = sys.stderr)

                is_better = len(hist_valid_scores) == 0 or valid_metric > max(hist_valid_scores)

                if is_better:
                    patience = 0
                    print("Save currently best model")
                    model.save(model_save_path)

                    torch.save(optimizer.state_dict(), model_save_path + '.optim')

                elif patience < max_patience:
                    patience += 1

                    print("hit patience %d" % patience, file = sys.stderr)
                    if patience == max_patience:
                        num_trial += 1

                        if num_trial == max_num_trial:
                            print("early stop!", file = sys.stderr)
                            exit(0)

                        #Learning rate decay
                        lr = optimizer.param_groups[0]['lr']* learning_rate_decay

                        #load previous best model
                        params = torch.load(model_save_path, map_location = lambda storage, loc: storage)

                        model.load_state_dict(params['state_dict'])

                        model = model.to(device)

                        optimizer.load_state_dict(torch.load(model_save_path + '.optim'))

                        #load learning rate
                        for param_group in optimizer.param_groups:
                            param_group['lr'] = lr

                        patience = 0

                if epoch == max_epoch:
                    print("maximum epoch reached!", file = sys.stderr)
                    exit(0)

            

            



def main(args):
    seed = args.seed
    torch.manual_seed(seed)
    if args.cuda:
        torch.cuda.manual_seed(seed)
    np.random.seed(seed**2)

    if args.mode == 'train':
        train(args)
    elif args.mode =='test':
        evaluate(args)
    else:
        raise RuntimeError('invalid run mode')
    



if __name__ == '__main__':
    args = get_args()
    logging.info('COMMAND: %s' % ' '.join(sys.argv))
    logging.info('Arguments: {}'.format(vars(args)))
    main(args)
