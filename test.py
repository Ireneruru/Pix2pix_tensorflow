from config import Config as conf
from data import *
import scipy.misc
from model import CGAN
from utils import imsave
import tensorflow as tf
import numpy as np
import time
from IPython import embed

def prepocess_test(img, cond):

    img = scipy.misc.imresize(img, [conf.train_size, conf.train_size])
    cond = scipy.misc.imresize(cond, [conf.train_size, conf.train_size])
    img = img.reshape(1, conf.img_size, conf.img_size, conf.img_channel)
    cond = cond.reshape(1, conf.img_size, conf.img_size, conf.img_channel)
    img = img/127.5 - 1.
    cond = cond/127.5 - 1.
    return img,cond

def test():

    if not os.path.exists("test"):
        os.makedirs("test")
    data = load_data()
    model = CGAN()

    d_opt = tf.train.AdamOptimizer(learning_rate=conf.learning_rate).minimize(model.d_loss, var_list=model.d_vars)
    g_opt = tf.train.AdamOptimizer(learning_rate=conf.learning_rate).minimize(model.g_loss, var_list=model.g_vars)

    saver = tf.train.Saver()

    counter = 0
    start_time = time.time()

    with tf.Session() as sess:
        saver.restore(sess, conf.model_path)
        test_data = data["train"]
        test_count = 0
        for img, cond in test_data:
            test_count += 1
            pimg, pcond = prepocess_test(img, cond)
            gen_img = sess.run(model.gen_img, feed_dict={model.image:pimg, model.cond:pcond})
	    embed()
            gen_img = gen_img.reshape(gen_img.shape[1:])
            gen_img = (gen_img + 1.) * 127.5
            image = np.concatenate((gen_img, cond), axis=1).astype(np.int)
	    embed()
            imsave(image, "./test" + "/%d.jpg" % test_count)

if __name__ == "__main__":
    test()
