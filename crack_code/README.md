### python：破解验证码
  本实验利用利用深度学习，来尝试破解验证吗。(参考网上内容)

**1. 数据生成**
```
usage: prepare_data.py [-h] [--image_num IMAGE_NUM] [--image_dir IMAGE_DIR]
                       [--tf_data_dir TF_DATA_DIR]
                       [--check_point_dir CHECK_POINT_DIR] [--gpu GPU]
                       [--learning_rate LEARNING_RATE]
                       [--batch_size BATCH_SIZE] [--epoch_size EPOCH_SIZE]
                       [--img_file IMG_FILE]

optional arguments:
  -h, --help            show this help message and exit
  --image_num IMAGE_NUM
                        生成验证图片数量
  --image_dir IMAGE_DIR
                        图片路径
  --tf_data_dir TF_DATA_DIR
                        tf_record文件路径
  --check_point_dir CHECK_POINT_DIR
                        checkpoint路径
  --gpu GPU             gpu
  --learning_rate LEARNING_RATE
                        学习率
  --batch_size BATCH_SIZE
                        batch size
  --epoch_size EPOCH_SIZE
                        epoch
  --img_file IMG_FILE   单图识别图片文件
python prepare_data.py --image_num 20000
```

**2. 训练过程**
```
usage: train.py [-h] [--image_num IMAGE_NUM] [--image_dir IMAGE_DIR]
                [--tf_data_dir TF_DATA_DIR]
                [--check_point_dir CHECK_POINT_DIR] [--gpu GPU]
                [--learning_rate LEARNING_RATE] [--batch_size BATCH_SIZE]
                [--epoch_size EPOCH_SIZE] [--img_file IMG_FILE]
python train.py
```


**3. 测试验证过程**
```
usage: test_one.py [-h] [--image_num IMAGE_NUM] [--image_dir IMAGE_DIR]
                [--tf_data_dir TF_DATA_DIR]
                [--check_point_dir CHECK_POINT_DIR] [--gpu GPU]
                [--learning_rate LEARNING_RATE] [--batch_size BATCH_SIZE]
                [--epoch_size EPOCH_SIZE] [--img_file IMG_FILE]
python test_one.py --img_file ./images/001y.jpg --batch_size 1

2018-12-13 23:10:00.440321: I tensorflow/core/platform/cpu_feature_guard.cc:141] Your CPU supports instructions that this TensorFlow binary was not compiled to use: SSE4.1 SSE4.2 AVX
2018-12-13 23:10:00.788711: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1392] Found device 0 with properties: 
name: GeForce GTX 1080 Ti major: 6 minor: 1 memoryClockRate(GHz): 1.582
pciBusID: 0000:02:00.0
totalMemory: 10.91GiB freeMemory: 10.45GiB
2018-12-13 23:10:00.788784: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1471] Adding visible gpu devices: 0
2018-12-13 23:10:01.133237: I tensorflow/core/common_runtime/gpu/gpu_device.cc:952] Device interconnect StreamExecutor with strength 1 edge matrix:
2018-12-13 23:10:01.133304: I tensorflow/core/common_runtime/gpu/gpu_device.cc:958]      0 
2018-12-13 23:10:01.133333: I tensorflow/core/common_runtime/gpu/gpu_device.cc:971] 0:   N 
2018-12-13 23:10:01.133780: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1084] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 5586 MB memory) -> physical GPU (device: 0, name: GeForce GTX 1080 Ti, pci bus id: 0000:02:00.0, compute capability: 6.1)
./checkpoint/crack_captcha.model-141300
predict: 0 0 1 y
```
