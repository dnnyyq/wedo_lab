# coding: utf-8
import tensorflow as tf
import numpy as np
import os
import argparse
from util import get_config, get_char_list, gen_code_image, set_env, check_mkdir
from gen_tfrecord_file import GenTfrecodFile
from nets import nets_factory
from PIL import Image

def load_check_point(checkpoint_dir, sess, saver):
    ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
    if ckpt and ckpt.model_checkpoint_path:
        ckpt_path = str(ckpt.model_checkpoint_path)
        print(ckpt_path)
        saver.restore(sess, ckpt_path)

def main(config):
    # init the env
    env_config = set_env(config.gpu)
    
    # check the path
    check_mkdir(config.image_dir)
    check_mkdir(config.tf_data_dir)
    check_mkdir(config.check_point_dir)
    
    # get the char list
    char_list, char_id_map, id_char_map, char_num = get_char_list()
    
    # train tfrecord
    train_tfrecord = '%s/%s' % (config.tf_data_dir, 'test.tfrecords')
    
    # input and output placeholder
    x = tf.placeholder(tf.float32, [None, 224, 224])  
    y0 = tf.placeholder(tf.float32, [None]) 
    y1 = tf.placeholder(tf.float32, [None]) 
    y2 = tf.placeholder(tf.float32, [None]) 
    y3 = tf.placeholder(tf.float32, [None])
    
    # get the input and putput data
    tfrecord_generator = GenTfrecodFile(config.image_dir, config.tf_data_dir, char_id_map, num_test=500, num_validate=500)
    image, image_raw, label0, label1, label2, label3 = tfrecord_generator.read_and_decode(train_tfrecord)
    image_batch, image_raw_batch,label_batch0, label_batch1, label_batch2, label_batch3 = tf.train.shuffle_batch(
        [image, image_raw,label0, label1, label2, label3], batch_size = config.batch_size,
        capacity = 200000, min_after_dequeue=10000, num_threads=1)
    print(image_batch[0].shape) 
    # the net
    train_network_fn = nets_factory.get_network_fn(
        'alexnet_v2',
         num_classes=char_num,
         weight_decay=0.0005,
        is_training=True)
    
    with tf.Session(config = env_config) as sess:
        # inputs: a tensor of size [batch_size, height, width, channels]
        X = tf.reshape(x, [config.batch_size, 224, 224, 1])
        # 数据输入网络得到输出值
        logits0,logits1,logits2,logits3,end_points = train_network_fn(X)
        
        predict0 = tf.reshape(logits0, [-1, char_num])  
        predict0 = tf.argmax(predict0, 1)  
        
        predict1 = tf.reshape(logits1, [-1, char_num])  
        predict1 = tf.argmax(predict1, 1)  
        
        predict2 = tf.reshape(logits2, [-1, char_num])  
        predict2 = tf.argmax(predict2, 1)  
        
        predict3 = tf.reshape(logits3, [-1, char_num])  
        predict3 = tf.argmax(predict3, 1)  
        
        # 初始化
        sess.run(tf.global_variables_initializer())
        # 载入训练好的模型
        saver = tf.train.Saver()
        
        #checkpoint = tf.train.latest_checkpoint(check_point_dir)
        #saver.restore(sess,checkpoint)
        load_check_point(config.check_point_dir, sess, saver)
        
        # 创建一个协调器，管理线程
        coord = tf.train.Coordinator()
        # 启动QueueRunner, 此时文件名队列已经进队
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        
        for i in range(10):
            # 获取一个批次的数据和标签
            b_image, b_image_raw, b_label0, b_label1 ,b_label2 ,b_label3 = sess.run([image_batch, 
                                                                        image_raw_batch, 
                                                                        label_batch0, 
                                                                        label_batch1, 
                                                                        label_batch2, 
                                                                        label_batch3])
            # 打印标签
            print('label:',id_char_map[b_label0[0]], id_char_map[b_label1[0]] ,id_char_map[b_label2[0]] ,id_char_map[b_label3[0]])
            # 预测
            label0,label1,label2,label3 = sess.run([predict0,predict1,predict2,predict3], feed_dict={x: b_image})
            # 打印预测值
            print('predict:',id_char_map[label0[0]],id_char_map[label1[0]],id_char_map[label2[0]],id_char_map[label3[0]]) 
                    
        # 通知其他线程关闭
        coord.request_stop() 

if __name__ == '__main__':
    config = get_config()
    main(config)

