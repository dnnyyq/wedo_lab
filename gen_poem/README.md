### python：为你写诗
  本实验利用利用深度学习，生成古典诗歌。
  
**1. 训练过程**
```
usage: train.py [-h] [--epoch_size EPOCH_SIZE] [--num_layers NUM_LAYERS]
                [--batch_size BATCH_SIZE] [--rnn_size RNN_SIZE]
                [--learning_rate LEARNING_RATE]
                [--checkpoint_dir CHECKPOINT_DIR] [--file_name FILE_NAME]
                [--start_word START_WORD]
python train.py
```


**2. 测试验证过程**
```
usage: test.py [-h] [--epoch_size EPOCH_SIZE] [--num_layers NUM_LAYERS]
               [--batch_size BATCH_SIZE] [--rnn_size RNN_SIZE]
               [--learning_rate LEARNING_RATE]
               [--checkpoint_dir CHECKPOINT_DIR] [--file_name FILE_NAME]
               [--start_word START_WORD]
python test.py --start_word "黄"              
```
