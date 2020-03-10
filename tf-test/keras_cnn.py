# coding:utf-8
from captcha.image import ImageCaptcha
import random
from PIL import Image
import numpy as np
import tensorflow as tf

# 验证码的内容是10个数字0~9，小写英文字母和大写英文字母，所以总的字符量为62种
# 生成的验证码都是带干扰线干扰点，并且各种颜色和变形处理都是有的
number = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
alphabet = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u',
            'v', 'w', 'x', 'y', 'z']
ALPHABET = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U',
            'V', 'W', 'X', 'Y', 'Z']
SAVE_PATH = "/Users/imbytecat/keras_cnn/"
CHAR_SET = number + alphabet + ALPHABET
CHAR_SET_LEN = len(CHAR_SET)
# 输入的图片为160 * 60的，灰度化预处理以后为一维数组，每张图片总共有9600个输入值
IMAGE_HEIGHT = 60
IMAGE_WIDTH = 160


# 使用captcha来生成验证码
def random_captcha_text(char_set=None, captcha_size=4):
    if char_set is None:
        char_set = number + alphabet + ALPHABET

    captcha_text = []
    for i in range(captcha_size):
        c = random.choice(char_set)
        captcha_text.append(c)
    return captcha_text


def gen_captcha_text_and_image(width=160, height=60, char_set=CHAR_SET):
    image = ImageCaptcha(width=width, height=height)

    captcha_text = random_captcha_text(char_set)
    captcha_text = ''.join(captcha_text)

    captcha = image.generate(captcha_text)

    # 将图片转成numpy数组
    captcha_image = Image.open(captcha)
    captcha_image = np.array(captcha_image)
    return captcha_text, captcha_image


text, image = gen_captcha_text_and_image(char_set=CHAR_SET)
MAX_CAPTCHA = len(text)
print('CHAR_SET_LEN=', CHAR_SET_LEN, ' MAX_CAPTCHA=', MAX_CAPTCHA)


# 根据分析，这类验证码的颜色对我们的识别没有影响，所以将图像进行预处理——灰度化（根据图片的不同可以做灰度化或者二值化等预处理操作）
def convert2gray(img):
    if len(img.shape) > 2:
        # 这里使用的是求均值的方法（正规的方法应该是RGB三个通道上按照一定的比例取值）
        gray = np.mean(img, -1)
        return gray
    else:
        return img


def text2vec(text):
    vector = np.zeros([MAX_CAPTCHA, CHAR_SET_LEN])
    for i, c in enumerate(text):
        idx = CHAR_SET.index(c)
        vector[i][idx] = 1.0
    return vector


def vec2text(vec):
    text = []
    for i, c in enumerate(vec):
        text.append(CHAR_SET[c])
    return "".join(text)


def get_next_batch(batch_size=128):
    batch_x = np.zeros([batch_size, IMAGE_HEIGHT, IMAGE_WIDTH, 1])
    # 输出的字符集有62个字符，并且每张图片有4位字符，总共有4 * 62 = 248个输出值（下面的batch_size为每批训练的图片数量）
    batch_y = np.zeros([batch_size, MAX_CAPTCHA, CHAR_SET_LEN])

    def wrap_gen_captcha_text_and_image():
        while True:
            text, image = gen_captcha_text_and_image(char_set=CHAR_SET)
            if image.shape == (60, 160, 3):
                return text, image

    for i in range(batch_size):
        text, image = wrap_gen_captcha_text_and_image()
        image = tf.reshape(convert2gray(image), (IMAGE_HEIGHT, IMAGE_WIDTH, 1))
        batch_x[i, :] = image
        batch_y[i, :] = text2vec(text)

    return batch_x, batch_y


def crack_captcha_cnn():
    model = tf.keras.Sequential()

    """
    输入层有9600个值，输出层有248个值，如果使用全连接层作为隐藏层则会需要天量的计算
    所以需要先使用卷积核池化操作尽可能的减少计算量（如果有一些深度学习基础的同学应该知道计算机视觉中一般都是用卷积升级网络来解决这类问题）
    图片像素不高，所以使用的卷积核和池大小不能太大，优先考虑3 * 3 和5 * 5 的卷积核，池大小使用2 * 2
    按照下面的神经网络模型，卷积池化以后的输出应该是128 * 17 * 5 = 10880（如果最后一层的深度仍然使用64的话，大小会减为一半）
    """

    model.add(tf.keras.layers.Conv2D(32, (3, 3)))
    model.add(tf.keras.layers.PReLU())
    model.add(tf.keras.layers.MaxPool2D((2, 2), strides=2))

    model.add(tf.keras.layers.Conv2D(64, (5, 5)))
    model.add(tf.keras.layers.PReLU())
    model.add(tf.keras.layers.MaxPool2D((2, 2), strides=2))

    model.add(tf.keras.layers.Conv2D(128, (5, 5)))
    model.add(tf.keras.layers.PReLU())
    model.add(tf.keras.layers.MaxPool2D((2, 2), strides=2))

    # 输出的的每一位的字符之间没有关联关系，所以仍然将输出值看成4组，需要将输出值调整为(4, 62)的数组
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(MAX_CAPTCHA * CHAR_SET_LEN))
    model.add(tf.keras.layers.Reshape([MAX_CAPTCHA, CHAR_SET_LEN]))
    # 识别的原理是计算每一位字符上某个字符出现的可能性最大，所以每张图片都是一个4位的多分类问题，最终输出使用softmax进行归一化
    model.add(tf.keras.layers.Softmax())

    return model


def train():
    try:
        model = tf.keras.models.load_model(SAVE_PATH + 'model')
    except Exception as e:
        print('#######Exception', e)
        model = crack_captcha_cnn()

    # 在tensorflow2.0中softmax对应的损失函数是categorical_crossentropy，按照这个来配置模型
    model.compile(optimizer='Adam',
                  metrics=['accuracy'],
                  loss='categorical_crossentropy')

    for times in range(500000):
        batch_x, batch_y = get_next_batch(512)
        print('times=', times, ' batch_x.shape=', batch_x.shape, ' batch_y.shape=', batch_y.shape)
        model.fit(batch_x, batch_y, epochs=4)
        print("y预测=\n", np.argmax(model.predict(batch_x), axis=2))
        print("y实际=\n", np.argmax(batch_y, axis=2))

        if 0 == times % 10:
            print("save model at times=", times)
            model.save(SAVE_PATH + 'model')


def predict():
    model = tf.keras.models.load_model(SAVE_PATH + 'model')
    success = 0
    count = 100
    for _ in range(count):
        data_x, data_y = get_next_batch(1)
        prediction_value = model.predict(data_x)
        data_y = vec2text(np.argmax(data_y, axis=2)[0])
        # 最后将可能性最大的那个下标取出，作为字符集的下标，获取实际对应的字符（当然我们在训练的时候没有必要转化为字符，直接下标比较一下是否正确就可以了）
        prediction_value = vec2text(np.argmax(prediction_value, axis=2)[0])

        if data_y.upper() == prediction_value.upper():
            print("y预测=", prediction_value, "y实际=", data_y, "预测成功。")
            success += 1
        else:
            print("y预测=", prediction_value, "y实际=", data_y, "预测失败。")

    print("预测", count, "次", "成功率 =", success / count)

    pass


if __name__ == "__main__":
    train()
    predict()
