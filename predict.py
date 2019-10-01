import re
import pickle
import argparse

import torch
from soynlp.tokenizer import LTokenizer

from utils import Params
from model.transformer import Transformer
from model.ops import create_target_mask, create_source_mask, create_subsequent_mask, create_non_pad_mask


def clean_text(text):
    """
    remove special characters from the input sentence to normalize it
    Args:
        text: (string) text string which may contain special character
    Returns:
        normalized sentence
    """
    text = re.sub('[-=+,#/\?:^$.@*\"※~&%ㆍ!』\\‘|\(\)\[\]\<\>`…》]', '', text)
    return text


def predict(config):
    params = Params('config/params.json')

    # load tokenizer and torchtext Fields
    pickle_tokenizer = open('pickles/tokenizer.pickle', 'rb')
    cohesion_scores = pickle.load(pickle_tokenizer)
    tokenizer = LTokenizer(scores=cohesion_scores)

    pickle_kor = open('pickles/kor.pickle', 'rb')
    kor = pickle.load(pickle_kor)

    pickle_eng = open('pickles/eng.pickle', 'rb')
    eng = pickle.load(pickle_eng)

    # select model and load trained model
    model = Transformer(params)

    model.load_state_dict(torch.load(params.save_model))
    model.to(params.device)
    model.eval()

    input = clean_text(config.input)

    # convert input into tensor and forward it through selected model
    tokenized = tokenizer.tokenize(input)
    indexed = [kor.vocab.stoi[token] for token in tokenized]

    source = torch.LongTensor(indexed).unsqueeze(0).to(params.device)  # [1, source length]: unsqueeze to add batch size
    target = torch.zeros(1, params.max_len).type_as(source.data)

    encoder_output = model.encoder(source)
    print(encoder_output)
    next_symbol = eng.vocab.stoi['<sos>']

    for i in range(0, params.max_len):
        target[0][i] = next_symbol
        dec_output = model.decoder(target, source, encoder_output)
        # dec_output = [1, target length, output dim]
        prob = dec_output.squeeze(0).max(dim=-1, keepdim=False)[1]
        next_word = prob.data[i]
        next_symbol = next_word.item()

    # translation_tensor = [target length] filed with word indices
    target = model(source, target)
    target = torch.argmax(target.squeeze(0), -1)
    print(target.shape)
    # target = target.squeeze(0).max(dim=-1, keepdim=False)
    translation = [eng.vocab.itos[token] for token in target][1:]

    translation = ' '.join(translation)
    print(f'kor> {config.input}')
    print(f'eng> {translation.capitalize()}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Kor-Eng Translation prediction')
    parser.add_argument('--input', type=str, default='오늘 우리는 맛있는 저녁을 먹을거야')
    config = parser.parse_args()

    predict(config)
