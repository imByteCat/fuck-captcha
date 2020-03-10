import os
import re
import random
import numpy as np
from PIL import Image
from config import CAPTCHA_LIST, CAPTCHA_LEN, CAPTCHA_HEIGHT, CAPTCHA_WIDTH, BASE_DIR, SAMPLE_DIR


def get_captcha_text_and_image(captcha_list):
    """
    :return: 验证码字符串，验证码图像 numpy 数组
    """
    captcha_index = random.randint(0, len(captcha_list) - 1)
    filepath = captcha_list[captcha_index][1]
    # 读取
    captcha_image = Image.open(filepath)
    # 转化为 numpy 数组
    captcha_image = np.array(captcha_image)
    return captcha_list[captcha_index][0], captcha_image


def get_captcha_list(path):
    captcha_list = []
    for entry in os.scandir(path):
        captcha = []
        captcha_match = re.match(r'(\d{2,6})_[0-9a-z]{32}\.[a-z]{3}', entry.name)
        if entry.is_file() and captcha_match:
            captcha.append(captcha_match.group(1))
            captcha.append(entry.path)
            captcha_list.append(captcha)
    return captcha_list


def convert2gray(numpy_image):
    """
    图片转为黑白，三维转一维
    :param numpy_image: numpy 对象
    :return:  灰度图 numpy
    """
    if len(numpy_image.shape) > 2:
        numpy_image = np.mean(numpy_image, -1)
    return numpy_image


def text2vec(text, captcha_len=CAPTCHA_LEN, captcha_list=CAPTCHA_LIST):
    """
    文本转为向量
    :param text:
    :param captcha_len:
    :param captcha_list:
    :return: vector 文本对应的向量形式
    """
    text_len = len(text)  # 欲生成验证码的字符长度
    if text_len > captcha_len:
        raise ValueError('给定的文本超出最大长度' + captcha_len)
    vector = np.zeros(captcha_len * len(captcha_list))  # 生成一个一维向量 验证码长度 * 字符列表长度
    for i in range(text_len):
        # 找到字符对应在字符列表中的下标值 + 字符列表长度 * i 的「一维向量」并赋值为 1
        vector[captcha_list.index(text[i]) + i * len(captcha_list)] = 1
    return vector


def vec2text(vec, captcha_list=CAPTCHA_LIST):
    """
    向量转为文本
    :param vec:
    :param captcha_list:
    :return: 向量的字符串形式
    """
    vec_idx = vec
    text_list = [captcha_list[int(v)] for v in vec_idx]
    return ''.join(text_list)


def get_next_batch(batch_count=60, width=CAPTCHA_WIDTH, height=CAPTCHA_HEIGHT):
    """
    获取训练图片组
    :param batch_count: default 60
    :param width: 验证码宽度
    :param height: 验证码高度
    :return: batch_x, batch_y
    """
    batch_x = np.zeros([batch_count, width * height])
    batch_y = np.zeros([batch_count, CAPTCHA_LEN * len(CAPTCHA_LIST)])
    for i in range(batch_count):  # 生成对应的训练集
        text, image = get_captcha_text_and_image(get_captcha_list(SAMPLE_DIR))
        image = convert2gray(image)  # 转灰度numpy
        # 将图片数组一维化 同时将文本也对应在两个二维组的同一行
        batch_x[i, :] = image.flatten() / 255
        batch_y[i, :] = text2vec(text)  # 验证码文本的向量形式
    # 返回该训练批次
    return batch_x, batch_y


if __name__ == '__main__':
    x, y = get_next_batch(batch_count=1)  # 默认为1用于测试集
    print(x, y)
