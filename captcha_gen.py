import random
import numpy as np
from PIL import Image
from captcha.image import ImageCaptcha

NUMBER = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
LOW_CASE = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u',
            'v', 'w', 'x', 'y', 'z']
UP_CASE = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U',
           'V', 'W', 'X', 'Y', 'Z']

CAPTCHA_LIST = NUMBER  # 定义验证码字符表
CAPTCHA_LEN = 4  # 验证码长度
CAPTCHA_HEIGHT = 70  # 验证码高度
CAPTCHA_WIDTH = 200  # 验证码宽度


def random_captcha_text(char_set=CAPTCHA_LIST, captcha_size=CAPTCHA_LEN):
    """
    随机生成定长字符串
    :param char_set: 备选字符串列表
    :param captcha_size: 字符串长度
    :return: 字符串
    """
    captcha_text = [random.choice(char_set) for _ in range(captcha_size)]
    return ''.join(captcha_text)


def get_captcha_text_and_image(width=CAPTCHA_WIDTH, height=CAPTCHA_HEIGHT):
    """
    生成随机验证码
    :param width: 验证码图片宽度
    :param height: 验证码图片高度
    :return: 验证码字符串，验证码图像np数组
    """
    captcha_image = ImageCaptcha(width=width, height=height)
    # 验证码文本
    captcha_text = random_captcha_text()
    # captcha = image.generate(captcha_text)
    # 保存
    captcha_filename = './sample/' + captcha_text + '.png'
    captcha_image.write(captcha_text, captcha_filename)
    # 读取
    captcha_image = Image.open(captcha_filename)
    # 转化为 numpy 数组
    captcha_image = np.array(captcha_image)
    return captcha_text, captcha_image


if __name__ == '__main__':
    text, image = get_captcha_text_and_image()
    print(text, image.shape)  # (70, 200, 3)
