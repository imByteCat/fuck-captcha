import tensorflow as tf
from model_train import cnn_graph
from captcha_gen import get_captcha_text_and_image
from util import vec2text, convert2gray
from util import CAPTCHA_LIST, CAPTCHA_WIDTH, CAPTCHA_HEIGHT, CAPTCHA_LEN
from PIL import Image


def captcha2text(image_list, height=CAPTCHA_HEIGHT, width=CAPTCHA_WIDTH):
    """
    验证码图片转化为文本
    :param image_list:
    :param height:
    :param width:
    :return:
    """
    x = tf.compat.v1.placeholder(tf.float32, [None, height * width])
    keep_prob = tf.compat.v1.placeholder(tf.float32)
    y_conv = cnn_graph(x, keep_prob, (height, width))
    saver = tf.compat.v1.train.Saver()
    with tf.compat.v1.Session() as sess:
        saver.restore(sess, tf.train.latest_checkpoint('model/'))
        predict = tf.argmax(input=tf.reshape(y_conv, [-1, CAPTCHA_LEN, len(CAPTCHA_LIST)]), axis=2)
        vector_list = sess.run(predict, feed_dict={x: image_list, keep_prob: 1})
        vector_list = vector_list.tolist()
        text_list = [vec2text(vector) for vector in vector_list]
        return text_list


if __name__ == '__main__':
    tf.compat.v1.disable_eager_execution()
    text, image = get_captcha_text_and_image()
    img = Image.fromarray(image)
    image = convert2gray(image)
    image = image.flatten() / 255
    pre_text = captcha2text([image])
    print("验证码正确值:", text, ' 模型预测值:', pre_text)
    img.show()
