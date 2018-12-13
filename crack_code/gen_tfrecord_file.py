# coding: utf-8
import tensorflow as tf
import os
import random
import math
import sys
from PIL import Image
import numpy as np


class GenTfrecodFile(object):
    """generate the tfrecord file"""
    def __init__(self, image_dir, tf_record_dir, char_id_map, num_test=500, num_validate=500):
    	self.image_dir = image_dir
    	self.tf_record_dir = tf_record_dir
    	self.char_id_map = char_id_map
    	self.num_test = num_test
    	self.num_validate = num_validate
    
    def int64_feature(self, values):
        if not isinstance(values, (tuple, list)):
            values = [values]
        return tf.train.Feature(int64_list=tf.train.Int64List(value=values))
 
    def bytes_feature(self, values):
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[values]))

    def _get_filenames_and_classes(self):
        photo_filenames = []
        for filename in os.listdir(self.image_dir):
            # get all files in the path
            path = os.path.join(self.image_dir, filename)
            photo_filenames.append(path)
        return photo_filenames
    
    def image_to_tfexample(self, image_data, label0, label1, label2, label3):
        #Abstract base class for protocol messages.
        return tf.train.Example(features=tf.train.Features(feature={
          'image': self.bytes_feature(image_data),
          'label0': self.int64_feature(label0),
          'label1': self.int64_feature(label1),
          'label2': self.int64_feature(label2),
          'label3': self.int64_feature(label3),
        }))
     
    #把数据转为TFRecord格式
    def _convert_char_code_dataset(self, split_name, filenames):
        assert split_name in ['train', 'test', 'validate']
     
        with tf.Session() as sess:
            #定义tfrecord文件的路径+名字
            output_filename = os.path.join(self.tf_record_dir, split_name + '.tfrecords')
            with tf.python_io.TFRecordWriter(output_filename) as tfrecord_writer:
                for i,filename in enumerate(filenames):
                    try: 
                        #读取图片
                        image_data = Image.open(filename)  
                        #根据模型的结构resize 为了配合alexnet
                        image_data = image_data.resize((224, 224))
                        #灰度化
                        image_data = np.array(image_data.convert('L'))
                        #将图片转化为bytes
                        image_data = image_data.tobytes()              
     
                        #获取label
                        labels = filename.split('/')[-1][0:4]
                        num_labels = []
                        for j in range(4):
                            num_labels.append(self.char_id_map[labels[j]])
                            #num_labels.append(int(labels[j]))
                                                
                        #生成protocol数据类型
                        example = self.image_to_tfexample(image_data, num_labels[0], num_labels[1], num_labels[2], num_labels[3])
                        tfrecord_writer.write(example.SerializeToString())
                        
                    except IOError as e:
                        print('Could not read:',filename)
                        print('Error:',e)
                        print('Skip it\n')
    
    def gen_tfrecord(self):
        # get the all image file
        photo_filenames = self._get_filenames_and_classes()
        
        # split data into train/test/validate set by random
        random.seed(0)
        random.shuffle(photo_filenames)
        training_filenames = photo_filenames[(self.num_test + self.num_validate):]
        testing_filenames = photo_filenames[:self.num_test]
        validating_filenames = photo_filenames[self.num_test:(self.num_test + self.num_validate)]
        
        # convert data into tfrecord file
        self._convert_char_code_dataset('train', training_filenames)
        self._convert_char_code_dataset('test', testing_filenames)
        self. _convert_char_code_dataset('validate', validating_filenames)  	

    def read_and_decode(self, filename):
        # new a file queue
        filename_queue = tf.train.string_input_producer([filename])
        reader = tf.TFRecordReader()
        # get the file
        _, serialized_example = reader.read(filename_queue)   
        features = tf.parse_single_example(serialized_example,
                                           features={
                                               'image' : tf.FixedLenFeature([], tf.string),
                                               'label0': tf.FixedLenFeature([], tf.int64),
                                               'label1': tf.FixedLenFeature([], tf.int64),
                                               'label2': tf.FixedLenFeature([], tf.int64),
                                               'label3': tf.FixedLenFeature([], tf.int64),
                                           })
        # get the image
        image = tf.decode_raw(features['image'], tf.uint8)
        
        # raw image
        image_raw = tf.reshape(image, [224, 224])
        
        # process the image data
        image = tf.cast(image_raw, tf.float32) / 255.0
        image = tf.subtract(image, 0.5)
        image = tf.multiply(image, 2.0)
        
        # get the label
        label0 = tf.cast(features['label0'], tf.int32)
        label1 = tf.cast(features['label1'], tf.int32)
        label2 = tf.cast(features['label2'], tf.int32)
        label3 = tf.cast(features['label3'], tf.int32)
     
        return image, image_raw, label0, label1, label2, label3
