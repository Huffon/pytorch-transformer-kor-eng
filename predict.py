import pickle
import argparse

import torch
from soynlp.tokenizer import LTokenizer

from utils import Params, clean_text, display_attention
from model.transformer import Transformer


def predict(config):
    input = clean_text(config.input)
    params = Params('config/params.json')

    # load tokenizer and torchtext Fields
    pickle_tokenizer = open('pickles/tokenizer.pickle', 'rb')
    cohesion_scores = pickle.load(pickle_tokenizer)
    tokenizer = LTokenizer(scores=cohesion_scores)

    pickle_kor = open('pickles/kor.pickle', 'rb')
    kor = pickle.load(pickle_kor)
    pickle_eng = open('pickles/eng.pickle', 'rb')
    eng = pickle.load(pickle_eng)
    eos_idx = eng.vocab.stoi['<eos>']
    print(eos_idx)
    print(eng.vocab.itos[eos_idx])

    # select model and load trained model
    model = Transformer(params)
    model.load_state_dict(torch.load(params.save_model))
    model.to(params.device)
    model.eval()

    # convert input into tensor and forward it through selected model
    tokenized = tokenizer.tokenize(input)
    indexed = [kor.vocab.stoi[token] for token in tokenized]

    source = torch.LongTensor(indexed).unsqueeze(0).to(params.device)  # [1, source_len]: unsqueeze to add batch size
    target = torch.zeros(1, params.max_len).type_as(source.data)       # [1, max_len]

    encoder_output = model.encoder(source)
    next_symbol = eng.vocab.stoi['<sos>']

    for i in range(0, params.max_len):
        if next_symbol == eos_idx:
            break
        target[0][i] = next_symbol
        print(target[0][i])
        decoder_output, _ = model.decoder(target, source, encoder_output)  # [1, target length, output dim]
        prob = decoder_output.squeeze(0).max(dim=-1, keepdim=False)[1]
        print(prob)
        next_word = prob.data[i]
        print(next_word)
        next_symbol = next_word.item()

    
    target[0][10] = 3
    print(target.shape)
    print(target[0][34])
    eos_idx = torch.where(target[0] == eos_idx)[0][0]
    eos_idx = eos_idx.item()
    print(eos_idx)
    target = target[0][:eos_idx].unsqueeze(0)

    # translation_tensor = [target length] filed with word indices
    target, attention_map = model(source, target)
    target = target.squeeze(0).max(dim=-1)[1]

    translated_token = [eng.vocab.itos[token] for token in target]
    print(translated_token)
    #translation = translated_token[:translated_token.index('<eos>')]
    #translation = ''.join(translation)
    translation = ''.join(translated_token)

    print(f'question> {config.input}')
    print(f'reply> {translation}')
    display_attention(tokenized, translated_token, attention_map[4].squeeze(0)[:-1])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Response Generation')
    parser.add_argument('--input', type=str, default='축구를 좋아하니')
    option = parser.parse_args()

    predict(option)
