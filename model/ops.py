import pickle
import numpy as np
import torch

pickle_eng = open('pickles/eng.pickle', 'rb')
eng = pickle.load(pickle_eng)
pad_idx = eng.vocab.stoi['<pad>']
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def create_subsequent_mask(target):
    # target = [batch size, target length]
    batch_size, target_length = target.size()
    '''
    if target length is 5 and diagonal is 1, this function returns
        [[0, 1, 1, 1, 1],
         [0, 0, 1, 1, 1],
         [0, 0, 0, 1, 1],
         [0, 0, 0, 1, 1],
         [0, 0, 0, 0, 1]]
    '''
    # torch.triu returns the upper triangular part of a matrix based on user defined diagonal
    subsequent_mask = torch.triu(torch.ones(target_length, target_length), diagonal=1).bool().to(device)
    # subsequent_mask = [target length, target length]

    # repeat subsequent mask 'batch size' times to cover all data instances in the batch
    subsequent_mask = subsequent_mask.unsqueeze(0).repeat(batch_size, 1, 1)
    # subsequent_mask = [batch size, target length, target length]

    return subsequent_mask


def create_source_mask(source):
    '''
    create masking tensor for encoder's self attention
    if sentence is [2, 193, 9, 27, 10003, 1, 1, 1, 3] and 2 denotes <sos>, 3 denotes <eos> and 1 denotes <pad>
    masking tensor will be [False, False, False, False, False, True, True, True, False]
    :param source: [batch size, source length]
    :return: source mask
    '''
    source_length = source.shape[1]

    # create boolean tensors which will be used to mask padding tokens of both source and target sentence
    source_mask = (source == pad_idx)
    # source_mask = [batch size, source length]

    # repeat sentence masking tensors 'sentence length' times
    source_mask = source_mask.unsqueeze(1).repeat(1, source_length, 1)
    # source_mask = [batch size, source length, source length]

    return source_mask


def create_target_mask(source, target, subsequent_mask):
    '''
    create masking tensor for decoder's self attention and decoder's attention on the output of encoder
    if sentence is [2, 193, 9, 27, 10003, 1, 1, 1, 3] and 2 denotes <sos>, 3 denotes <eos> and 1 denotes <pad>
    masking tensor will be [False, False, False, False, False, True, True, True, False]
    :param source: [batch size, source length]
    :param target: [batch size, target length]
    :param subsequent_mask: [batch size, target length, target length]
    :return:
    '''
    target_length = target.shape[1]

    # create boolean tensors which will be used to mask padding tokens of both source and target sentence
    source_mask = (source == pad_idx)
    target_mask = (target == pad_idx)
    # target_mask    = [batch size, target length]

    # repeat sentence masking tensors 'sentence length' times
    dec_enc_mask = source_mask.unsqueeze(1).repeat(1, target_length, 1)
    target_mask = target_mask.unsqueeze(1).repeat(1, target_length, 1)

    # dec_enc_mask   = [batch size, target length, source length]
    # target_mask    = [batch size, target length, target length]

    # combine <pad> token masking tensor and subsequent masking tensor for decoder's self attention
    target_mask = target_mask | subsequent_mask
    # target_mask = [batch size, target length, target length]

    return target_mask, dec_enc_mask


def create_non_pad_mask(sentence):
    '''
    create non-pad masking tensor which will be used to extract non-padded tokens from output
    if sentence is [2, 193, 9, 27, 1, 1, 1, 3]
    this function returns [[1], [1], [1], [1], [0], [0], [0], [1]]
    '''
    return sentence.ne(pad_idx).type(torch.float).unsqueeze(-1)


def create_positional_encoding(batch_size, sentence_len, hidden_dim):
    # PE(pos, 2i)     = sin(pos/10000 ** (2*i / hidden_dim))
    # PE(pos, 2i + 1) = cos(pos/10000 ** (2*i / hidden_dim))
    sinusoid_table = np.array([pos / np.power(10000, 2 * i / hidden_dim)
                               for pos in range(sentence_len) for i in range(hidden_dim)])
    # sinusoid_table = [sentence length * hidden dim]

    sinusoid_table = sinusoid_table.reshape(sentence_len, -1)
    # sinusoid_table = [sentence length, hidden dim]

    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # calculate pe for even dimension
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # calculate pe for odd dimension

    # convert numpy based sinusoid table to torch.tensor and repeat it 'batch size' times
    sinusoid_table = torch.FloatTensor(sinusoid_table).to(device)
    sinusoid_table = sinusoid_table.unsqueeze(0).repeat(batch_size, 1, 1)
    # sinusoid_table = [batch size, sentence length, hidden dim]

    return sinusoid_table
