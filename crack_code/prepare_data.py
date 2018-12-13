# coding: utf-8
from captcha.image import ImageCaptcha
import numpy as np
import os
from util import get_config, get_char_list, gen_code_image, set_env, check_mkdir
from gen_tfrecord_file import GenTfrecodFile


def main(config):
    # init the env
    env_config = set_env(config.gpu)
    
    # get the char list
    char_list, char_id_map, id_char_map, char_num = get_char_list()
    # check the path
    check_mkdir(config.image_dir)
    check_mkdir(config.tf_data_dir)
    check_mkdir(config.check_point_dir)
    
    # generate the code images
    for i in range(config.image_num):
        gen_code_image(char_list, config.image_dir, 4)
    
    # generate the tfrecord file
    tfrecord_generator = GenTfrecodFile(config.image_dir, config.tf_data_dir, char_id_map, num_test=500, num_validate=500)
    tfrecord_generator.gen_tfrecord()  	
	
if __name__ == '__main__':
    config = get_config()
    main(config)
