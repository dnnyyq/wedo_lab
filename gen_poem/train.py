# coding: utf-8
import tensorflow as tf
import numpy as np
import os
import argparse
from char_rnn import CharRNN
from util import process_poems, generate_batch, pretify_poem

start_token = 'S'
end_token = 'E'

def get_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch_size', type=int, default='20000') 
    parser.add_argument('--num_layers', type=int, default='2')
    parser.add_argument('--batch_size', type=int, default='64')
    parser.add_argument('--rnn_size', type=int, default='128')
    parser.add_argument('--learning_rate', type=float, default='0.001')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoint')
    parser.add_argument('--file_name', type=str, default='./data/poetryTang.txt')
    parser.add_argument('--start_word', type=str, default='´º')
    config = parser.parse_args()
    return config	

def main(config):
    words, word_id_map, poems_id_vector, id_word_map = process_poems(config.file_name, start_token='S', end_token='E')
    generate_batches = generate_batch(config.batch_size, poems_id_vector, word_id_map)
    with tf.Session() as sess:
        model = CharRNN(sess, config.epoch_size, config.num_layers, config.batch_size, config.learning_rate,
                    len(words)+1, config.rnn_size, generate_batches, config.checkpoint_dir, False)
        model.train()

if __name__ == '__main__':
    config = get_config()
    main(config)


