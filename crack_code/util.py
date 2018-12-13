# coding: utf-8
from captcha.image import ImageCaptcha
import numpy as np
import os
import argparse
import tensorflow as tf

def gen_code_image(char_list, image_dir, text_num=4):
    captcha_text = []
    for i in range(text_num):
        random_text = np.random.choice(char_list)
        captcha_text.append(random_text)
    # print(captcha_text)
    captcha_texts = ''.join(captcha_text)
    image = ImageCaptcha()
    captcha = image.generate(captcha_texts)
    image_path = image_dir
    image.write(captcha_texts, image_path + captcha_texts + '.jpg')

def check_mkdir(path):
    if not os.path.isdir(path):
        os.makedirs(path)

def set_env(gpu):
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu   #指定第一块GPU可用
    env_config = tf.ConfigProto()
    env_config.gpu_options.per_process_gpu_memory_fraction = 0.5  # 程序最多只能占用指定gpu50%的显存
    env_config.gpu_options.allow_growth = True      #程序按需申请内存
    return env_config	

def get_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_num', type=int, default='200000', help='生成验证图片数量') 
    parser.add_argument('--image_dir', type=str, default='./images/', help='图片路径')
    parser.add_argument('--tf_data_dir', type=str, default='./data/', help='tf_record文件路径')
    parser.add_argument('--check_point_dir', type=str, default='./checkpoint', help='checkpoint路径')
    parser.add_argument('--gpu', type=str, default='0', help='gpu')
    parser.add_argument('--learning_rate', type=float, default='0.0001', help='学习率')
    parser.add_argument('--batch_size', type=int, default='25', help='batch size')
    parser.add_argument('--epoch_size', type=int, default='200001', help='epoch')
    parser.add_argument('--img_file', type=str, default='./images/82xm.jpg', help='单图识别图片文件')
    
    config = parser.parse_args()
    return config
    

def get_char_list():
    number = ['0','1','2','3','4','5','6','7','8','9']
    alphabet = ['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z']
    ALPHABET = ['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z']
    char_list = alphabet + number
    char_id_map = dict(zip(char_list, range(len(char_list))))
    id_char_map = dict(zip(range(len(char_list)), char_list))
    char_num = len(char_list)
    return char_list, char_id_map, id_char_map, char_num
 

    
