# coding: utf-8
import tensorflow as tf
import os
import numpy as np
class CharRNN(object):
    def __init__(self, sess, epoch_size, num_layers, batch_size, learning_rate
                 , num_classes, rnn_size, generate_batch, checkpoint_dir, is_test):
        self.sess = sess
        self.generate_batch = generate_batch
        self.num_classes = num_classes
        self.epoch_size = epoch_size
        self.rnn_size = rnn_size
        self.num_layers = num_layers
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.checkpoint_dir = checkpoint_dir
        self.is_test = is_test
        if is_test:
            self.batch_size = 1
        self.bulid_net()
        self.saver = tf.train.Saver(max_to_keep=4)
        
    def bulid_net(self):
        ## 确定输入和输出站位信息
        with tf.variable_scope('input', reuse=tf.AUTO_REUSE):
            self.x = tf.placeholder(tf.int32, [self.batch_size, None]) 
            self.y_ = tf.placeholder(tf.int32, [self.batch_size, None]) 
            self.prob = tf.placeholder(tf.float16, name='keep_prob')
            with tf.device('/cpu:0'):
                embedding = tf.get_variable('embedding', 
                                            initializer=tf.random_uniform([self.num_classes, self.rnn_size]
                                                                          , -1.0, 1.0))
                self.inputs = tf.nn.embedding_lookup(embedding, self.x)
                
        with tf.variable_scope('net', reuse=tf.AUTO_REUSE):
            one_cell = tf.contrib.rnn.BasicLSTMCell(self.rnn_size)
            multi_cell = tf.contrib.rnn.MultiRNNCell( [one_cell] * self.num_layers, state_is_tuple=True)
            self.initial_state = multi_cell.zero_state(self.batch_size, tf.float32)
            outputs, self.last_state = tf.nn.dynamic_rnn(multi_cell, self.inputs, initial_state=self.initial_state)
            output = tf.reshape(outputs,[-1, self.rnn_size])
        
        with tf.variable_scope('softmax', reuse=tf.AUTO_REUSE):
            softmax_w = tf.Variable(tf.truncated_normal([self.rnn_size, self.num_classes], stddev=0.1))
            softmax_b = tf.Variable(tf.zeros(self.num_classes))
            self.logits = tf.nn.bias_add(tf.matmul(output, softmax_w), bias=softmax_b)
            self.prediction = tf.nn.softmax(self.logits)
            
        with tf.variable_scope('opt', reuse=tf.AUTO_REUSE):
            y_one_hot = tf.one_hot(tf.reshape(self.y_, [-1]), self.num_classes)
            loss = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=y_one_hot)
            self.loss = tf.reduce_mean(loss)
            tf.summary.scalar('loss',self.loss)
        self.train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)
    
    def train(self):
        init = tf.global_variables_initializer()
        self.sess.run(init)
        self.load_check_point()
        merged = tf.summary.merge_all()
        writer = tf.summary.FileWriter('logs',self.sess.graph)
        for i in range(self.epoch_size):
            for input_x, output_y in self.generate_batch:
                feed_dict = {self.x : input_x, self.y_ : output_y}
                loss, _, _ = self.sess.run([self.loss, self.train_op, self.last_state], feed_dict=feed_dict)
            if i % 50 == 0:
                print('epoch: %s, loss : %s' % (i, loss))
                self.saver.save(self.sess, os.path.join(self.checkpoint_dir, 'model'), global_step=i)
                result = self.sess.run(merged, feed_dict = feed_dict)
                writer.add_summary(result,i)
                
    def load_check_point(self):
        ckpt = tf.train.get_checkpoint_state(self.checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_path = str(ckpt.model_checkpoint_path)
            print(ckpt_path)
            self.saver.restore(self.sess, ckpt_path)
            
    def id_to_word(self, prediction, id_word_map):
        t = np.cumsum(prediction)
        s = np.sum(prediction)
        coff = np.random.rand(1)
        index = int(np.searchsorted(t, coff * s))
        #sample = np.argmax(prediction)
        #if sample > len(id_word_map):
        #    sample = len(id_word_map) - 1
        return id_word_map[index]
    
    def gen_poem(self, start_word, word_int_map, start_token, end_token, id_word_map):
        init = tf.global_variables_initializer()
        self.sess.run(init)
        self.load_check_point()
        x = np.array([list(map(word_int_map.get, start_token))])
        [prediction,last_state] = self.sess.run([self.prediction, self.last_state], feed_dict={self.x : x})
        poem = ''
        if start_word :
            word = start_word
        else:
            word = self.id_to_word(prediction, id_word_map)
        while word != end_token:
            poem += word
            x = np.zeros((1, 1))
            x[0,0] = word_int_map[word]
            [prediction, last_state] = self.sess.run([self.prediction, self.last_state],
                                             feed_dict={self.x: x, self.initial_state: last_state})
            word = self.id_to_word(prediction, id_word_map)
        return poem

