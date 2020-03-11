import os

NUMBER = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
LOW_CASE = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u',
            'v', 'w', 'x', 'y', 'z']
UP_CASE = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U',
           'V', 'W', 'X', 'Y', 'Z']

CHARSET_LIST = NUMBER  # 定义验证码字符表
CAPTCHA_LEN = 2  # 验证码长度
CAPTCHA_HEIGHT = 70  # 验证码高度
CAPTCHA_WIDTH = 200  # 验证码宽度

BASE_DIR = os.path.split(os.path.realpath(__file__))[0]
SAMPLE_DIR = os.path.join(BASE_DIR, "sample")
MODEL_DIR = os.path.join(BASE_DIR, "model")
PRESET_ACCURACY = 0.95  # 预设模型准确率标准，默认 0.95
FINAL_ACCURACY = 0.99  # 达到一定准确率就退出训练，默认 0.99
