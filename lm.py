import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence


class RNNLM(nn.Module):
    """ 
    Language Model with basic LSTM
    """

    def __init__(self, embed_size, hidden_size, vocab, dropout_rate = 0.1, device = 'cpu', tie_embed = False):
        super(RNNLM, self).__init__()

        self.vocab = vocab
        self.device = device
        vocab_size = len(vocab)
        self.dropout_rate = dropout_rate
        self.hidden_size = hidden_size
        self.embed = nn.Embedding(vocab_size, embed_size, padding_idx = self.vocab.word2id['<pad>'])
        self.dropout = nn.Dropout(dropout_rate)
        self.rnn = nn.LSTMCell(embed_size + hidden_size, hidden_size, bias = True)
        self.dec_projection = nn.Linear(hidden_size,hidden_size, bias = True)
        self.vocab_projection = nn.Linear(hidden_size, vocab_size)

        #Use Tied-embedding
        if tie_embed: 
            assert embed_size == hidden_size

            self.embed.weight = self.vocab_projection.weight

    def forward(self, sents):
        """
        Performs a forward-propagration and calulate per-token-cross-entropy-loss

        param  sents: input batches (len, batch_size)

        rtype  scores: loss
        """
        sent_padded = self.vocab.to_input_tensor(sents, device = self.device)
        lm_output = self.lm(sent_padded)
        lm_vocab = self.vocab_projection(lm_output) # Final projection to vocab size ( batch, V)
        P = F.log_softmax(lm_vocab, dim = -1)

        target_masks = (sent_padded != self.vocab.word2id['<pad>']).float()
        
        score = torch.gather(P,index = sent_padded[1:].unsqueeze(-1),dim = -1).squeeze(-1)* target_masks[1:]
        scores = score.sum(dim = 0)
        return scores

    def lm(self, src):
        """
        Language Model prediction. Previous time-step prediction is added  to current step embedding 
        to make model aware of previous choice. 

        param  src: input sentences (max_len, batch_size)
        rtype  prediction for each time step (batch_size, embed_size + hidden_size)
        """

        src = src[:-1]
        batch_size = src.size(1)
        embed = self.embed(src) # (max_len , batch_size, embed_size)
        dec_state = (torch.zeros(batch_size,self.hidden_size, device = self.device), torch.zeros(batch_size, self.hidden_size, device = self.device))
        o_prev = torch.zeros(batch_size,self.hidden_size, device = self.device)
        combined_outputs = []
        for y_t in torch.split(embed,1,0): #iterate over each time step
            y_squeeze = torch.squeeze(y_t,0)
            y_bar = torch.cat((y_squeeze,o_prev),1)
            dec_state, combined_output = self.predict(y_bar,dec_state)
            combined_outputs.append(combined_output)
            o_prev = combined_output

        combined_outputs = torch.stack(combined_outputs,0)

        return combined_outputs

    def predict(self,y_bar,dec_state):
        """
        """
        dec_hidden, dec_cell = self.rnn(y_bar,dec_state)
        dec_state = (dec_hidden,dec_cell)
        dec_proj = self.dec_projection(dec_hidden)
        output = self.dropout(torch.tanh(dec_hidden))

        return dec_state, output


    @staticmethod
    def load(model_path):
        params = torch.load(model_path, map_location = lambda storage, loc: storage)
        args = params['args']
        model = RNNLM(vocab = params['vocab'], **args)
        model.load_state_dict(params['state_dict'])

        return model

    def save(self, path):
        print('save model to [%s]' % path, file = sys.stderr)
        params = {
                'args': dict(embed_size = self.embed, hidden_size = self.hidden_size, dropout_rate = self.dropout_rate,device = self.device),
                'vocab': self.vocab,
                'state_dict': self.state_dict()
                }
        torch.save(params, path)


