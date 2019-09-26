## Transformer PyTorch implementation
This repo contains Transformer implementation used to **translate Korean sentence into English sentence**.

I used translation dataset, but you can apply these models to any sequence to sequence (i.e. text generation) tasks such as text summarization, response generation, ..., etc.

All of base codes are based on this great [**seq2seq tutorial**](https://github.com/bentrevett/pytorch-seq2seq).

In this project, I specially used Korean-English translation corpus from [**AI Hub**](http://www.aihub.or.kr/) to apply torchtext into Korean dataset. 

I can not upload the used dataset because it requires an approval from AI Hub. You can get an approval from AI Hub, if you request it to admins.

And I also used [**soynlp**](https://github.com/lovit/soynlp) library which is used to tokenize Korean sentence. 
It is really nice and easy to use, you should try if you handle Korean sentences :)

<br/>

### Overview
- Number of train data: 75,000
- Number of validation data: 10,000
- Number of test data: 10,000
```
Example: 
{
  'kor': '['부러진', '날개로', '다시한번', '날개짓을', '하라']',
  'eng': '['wings', 'once', 'again', 'with', 'broken', 'wings']'
}
```
<br/>

### Requirements

- Following libraries are fundamental to this repo. Since I used conda environment `requirements.txt` has much more dependent libraries. 
- If you encounters any dependency problem, just use following command 
    - `pip install -r requirements.txt`

```
en-core-web-sm==2.1.0
matplotlib==3.1.1
numpy==1.16.4
pandas==0.25.1
scikit-learn==0.21.3
soynlp==0.0.493
spacy==2.1.8
torch==1.2.0
torchtext==0.4.0
```
<br/>


### Usage
- Before training the model, you should train `soynlp tokenizer` on your training dataset and build vocabulary using following code. 
- You can determine the size of vocabulary of Korean and English dataset. 
- In general, Korean dataset creates the larger size vocabulary than English dataset. Therefore to make balance, you have to pick **proper** vocab size
- By running following code, you will get `tokenizer.pickle`, `kor.pickle` and `eng.pickle` which are used to train, 
test the model and predict user's input sentence

```
python build_pickle.py --kor_vocab KOREAN_VOCAB_SIZE --eng_vocab ENGLISH_VOCAB_SIZE
```


- For training, run `main.py` with train mode (which is default option)

```
python main.py --model MODEL_NAME
```

- For testing, run `main.py` with test mode

```
python main.py --model MODEL_NAME --mode test
```

- For predicting, run `predict.py` with your Korean input sentence. 
- *Don't forget to wrap your input with double quotation mark !*

```
python predict.py --model MODEL_NAME --input "YOUR_KOREAN_INPUT"
```
