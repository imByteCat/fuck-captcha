import tensorflow as tf
from model_train import cnn_graph
from util import get_captcha_text_and_image, get_captcha_list
from util import vec2text, convert2gray
from config import CHARSET_LIST, CAPTCHA_WIDTH, CAPTCHA_HEIGHT, CAPTCHA_LEN, MODEL_DIR, SAMPLE_DIR
from PIL import Image


def captcha2text(image_list, height=CAPTCHA_HEIGHT, width=CAPTCHA_WIDTH):
    """
    验证码图片转化为文本
    :param image_list:
    :param height:
    :param width:
    :return:
    """
    tf.compat.v1.reset_default_graph()
    x = tf.compat.v1.placeholder(tf.float32, [None, height * width])
    keep_prob = tf.compat.v1.placeholder(tf.float32)
    y_conv = cnn_graph(x, keep_prob, (height, width))
    saver = tf.compat.v1.train.Saver()
    with tf.compat.v1.Session() as sess:
        saver.restore(sess, tf.train.latest_checkpoint(MODEL_DIR))
        predict = tf.argmax(input=tf.reshape(y_conv, [-1, CAPTCHA_LEN, len(CHARSET_LIST)]), axis=2)
        vector_list = sess.run(predict, feed_dict={x: image_list, keep_prob: 1})
        vector_list = vector_list.tolist()
        text_list = [vec2text(vector) for vector in vector_list]
        return text_list


if __name__ == '__main__':
    tf.compat.v1.disable_eager_execution()
    TEST_NUMBER = int(input("Test number: "))
    success_number = 0
    for test in range(TEST_NUMBER):
        text, image = get_captcha_text_and_image(get_captcha_list(SAMPLE_DIR))
        # img = Image.fromarray(image)
        # img.show()
        image = convert2gray(image)
        image = image.flatten() / 255
        predict_text = captcha2text([image])[0]
        print("验证码正确值：%s 模型预测值：%s" % (text, predict_text))
        if predict_text == text:
            success_number += 1
    print("本次测试准确率：%.2f" % (success_number / TEST_NUMBER))
