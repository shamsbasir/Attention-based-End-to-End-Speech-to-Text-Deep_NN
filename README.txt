This is part of homework 4 for CMU11785. The objective is to generate transcript of speech with attention mechanism. 


Data : https://www.kaggle.com/c/11-785-fall-20-homework-4-part-2/data
```
Model : 
Seq2Seq(
  (encoder): Encoder(
    (lstm): LSTM(40, 256, batch_first=True, bidirectional=True)
    (pBLSTM_block): ModuleList(
      (0): pBLSTM(
        (dropout): LockedDropout()
        (blstm): LSTM(1024, 256, batch_first=True, bidirectional=True)
      )
      (1): pBLSTM(
        (dropout): LockedDropout()
        (blstm): LSTM(1024, 256, batch_first=True, bidirectional=True)
      )
    )
    (key_network): Linear(in_features=512, out_features=128, bias=True)
    (value_network): Linear(in_features=512, out_features=128, bias=True)
  )
  (decoder): Decoder(
    (embedding): Embedding(35, 256, padding_idx=0)
    (lstm1): LSTMCell(384, 512)
    (lstm2): LSTMCell(512, 128)
    (attention): Attention()
    (character_prob): Linear(in_features=256, out_features=35, bias=True)
  )
)

```
Params : 
train batch size 	: 256
validation batch size 	: 64
testing batch size 	: 1 
learning rate 		: 0.01 
teacher forcing : starts with 10% and increases 1% per epochs


optimizer     : Adams 
Loss function : CrossEntropy
```

directory structure : 
|__checkpoint 
|__ main.py
```
Note : Data path should be updated inside hw4p2.py


