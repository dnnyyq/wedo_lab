# coding: utf-8
import numpy as np
import collections
import os
import codecs
import traceback

def process_poems(file_name, start_token='S', end_token='E'):
    poems_data = []
    #file_name = '/cache5/wedo/Deep-Learning-21-Examples-master/chapter_12/data/poetryTang.txt'
    with codecs.open(file_name, 'r', 'utf-8') as f:
        for line in f.readlines():
            try:
                title, author, content = line.strip().split('::')
                content = content.replace(' ','')
                if '_' in content or '(' in content or '《' in content or '[' in content or \
                        start_token in content or end_token in content:
                    continue
                if len(content) < 5 or len(content) > 79:
                    continue
                poem = '%s%s%s' % (start_token, content, end_token)
                poems_data.append(poem)
            except ValueError as e:
                print('%s' % traceback.format_exc())
    poems_data = sorted(poems_data, key=lambda line : len(line))
    all_words = []
    for poem in poems_data:
        all_words += [word for word in poem]
    words_counter = collections.Counter(all_words)
    ## 按照字符频率排序
    count_pairs = sorted(words_counter.items(), key=lambda x: -x[1])
    words, _ = zip(*count_pairs)

    ## 为字符进行编码
    ## 添加空格元组元素
    words = words[:len(words)] + (' ',)
    word_id_map = dict(zip(words, range(len(words))))
    poems_id_vector = [list(map(word_id_map.get, poem)) for poem in poems_data]
    id_word_map = dict(zip(range(len(words)), words))
    return words, word_id_map, poems_id_vector, id_word_map

## 生成batch 
def generate_batch(batch_size, poems_id_vector, word_id_map):
    chunks = len(poems_id_vector)//batch_size
    print(chunks)
    for i in range(chunks):
        batch_begin_index = i * batch_size
        batch_end_index = (i + 1) * batch_size
        batches = poems_id_vector[batch_begin_index:batch_end_index]
        ## 找出batchs 中字数最多的，对其他的样本进行补空格操作，使得所有的batch 样本统一长度
        length = max(map(len, batches))
        x_data = np.full((batch_size, length), word_id_map[' '], np.int32)
        for j in range(batch_size):
            x_data[j, :len(batches[j])] = batches[j]
        y_data = np.copy(x_data)
        y_data[:, :-1] = x_data[:, 1:]
        yield x_data, y_data

def pretify_poem(poem_word):
    poem = poem_word.strip().split('。')
    for i in range(len(poem)):
        print(poem[i])
