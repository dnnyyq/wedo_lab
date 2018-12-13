# coding: utf-8
import tensorflow as tf
import numpy as np
import os
import argparse
from util import get_config, get_char_list, gen_code_image, set_env, check_mkdir
from gen_tfrecord_file import GenTfrecodFile
from nets import nets_factory
from PIL import Image

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
    train_tfrecord = '%s/%s' % (config.tf_data_dir, 'train.tfrecords')
    
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
        
        # 把标签转成one_hot的形式
        one_hot_labels0 = tf.one_hot(indices=tf.cast(y0, tf.int32), depth=char_num)
        one_hot_labels1 = tf.one_hot(indices=tf.cast(y1, tf.int32), depth=char_num)
        one_hot_labels2 = tf.one_hot(indices=tf.cast(y2, tf.int32), depth=char_num)
        one_hot_labels3 = tf.one_hot(indices=tf.cast(y3, tf.int32), depth=char_num)
        
        # 计算loss
        loss0 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits0,labels=one_hot_labels0)) 
        loss1 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits1,labels=one_hot_labels1)) 
        loss2 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits2,labels=one_hot_labels2)) 
        loss3 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits3,labels=one_hot_labels3)) 
        # 计算总的loss
        total_loss = (loss0+loss1+loss2+loss3)/4.0
        # 优化total_loss
        optimizer = tf.train.AdamOptimizer(learning_rate=config.learning_rate).minimize(total_loss) 
        
        # 计算准确率
        correct_prediction0 = tf.equal(tf.argmax(one_hot_labels0,1),tf.argmax(logits0,1))
        accuracy0 = tf.reduce_mean(tf.cast(correct_prediction0,tf.float32))
        
        correct_prediction1 = tf.equal(tf.argmax(one_hot_labels1,1),tf.argmax(logits1,1))
        accuracy1 = tf.reduce_mean(tf.cast(correct_prediction1,tf.float32))
        
        correct_prediction2 = tf.equal(tf.argmax(one_hot_labels2,1),tf.argmax(logits2,1))
        accuracy2 = tf.reduce_mean(tf.cast(correct_prediction2,tf.float32))
        
        correct_prediction3 = tf.equal(tf.argmax(one_hot_labels3,1),tf.argmax(logits3,1))
        accuracy3 = tf.reduce_mean(tf.cast(correct_prediction3,tf.float32)) 
        
        # 用于保存模型
        saver = tf.train.Saver(max_to_keep=4)
        # 初始化
        sess.run(tf.global_variables_initializer())
        
        # 创建一个协调器，管理线程
        coord = tf.train.Coordinator()
        # 启动QueueRunner, 此时文件名队列已经进队
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        
        
        for i in range(config.epoch_size):
            b_image, b_image_raw, b_label0, b_label1 ,b_label2 ,b_label3 = sess.run([image_batch, 
                                                                        image_raw_batch, 
                                                                        label_batch0, 
                                                                        label_batch1, 
                                                                        label_batch2, 
                                                                        label_batch3])
            # 优化模型
            sess.run(optimizer, feed_dict={x: b_image, y0:b_label0, y1: b_label1, y2: b_label2, y3: b_label3})  
     
            # 每迭代20次计算一次loss和准确率  
            if i % 20 == 0:  
                acc0,acc1,acc2,acc3,loss_ = sess.run([accuracy0,accuracy1,accuracy2,accuracy3,total_loss],feed_dict={x: b_image,
                                                                                                                    y0: b_label0,
                                                                                                                    y1: b_label1,
                                                                                                                    y2: b_label2,
                                                                                                                    y3: b_label3}) 
                print ("Iter:%d  Loss:%.3f  Accuracy:%.2f,%.2f,%.2f,%.2f  Learning_rate:%.6f" % (i,loss_,acc0,acc1,acc2,acc3,config.learning_rate))
                 
                # 保存模型	
                if i % 300 == 0:
                    saver.save(sess, config.check_point_dir, global_step=i)
                if acc0 > 0.90 and acc1 > 0.90 and acc2 > 0.90 and acc3 > 0.90: 
                	break
                    
        # 通知其他线程关闭
        coord.request_stop()
        # 其他所有线程关闭之后，这一函数才能返回
        coord.join(threads)   

if __name__ == '__main__':
    config = get_config()
    main(config)
