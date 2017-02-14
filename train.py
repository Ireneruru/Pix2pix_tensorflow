from config import Config as conf
from data import *
from model import CGAN
from utils import imsave
import tensorflow as tf
import numpy as np
import time

def prepocess_train(img, cond):
    img = img.reshape(1, conf.img_size, conf.img_size, conf.img_channel)
    cond = cond.reshape(1, conf.img_size, conf.img_size, conf.img_channel)
    img = img/127.5 - 1.
    cond = cond/127.5 - 1.
    return img,cond

def prepocess_test(img, cond):
    img = img.reshape(1, conf.img_size, conf.img_size, conf.img_channel)
    cond = cond.reshape(1, conf.img_size, conf.img_size, conf.img_channel)
    img = img/127.5 - 1.
    cond = cond/127.5 - 1.
    return img,cond

def train():
    data = load_data()
    model = CGAN()

    d_opt = tf.train.AdamOptimizer(learning_rate=conf.learning_rate).minimize(model.d_loss)
    g_opt = tf.train.AdamOptimizer(learning_rate=conf.learning_rate).minimize(model.g_loss)

    saver = tf.train.Saver()

    counter = 0
    start_time = time.time()

    with tf.Session() as sess:
        if conf.model_path == "":
            sess.run(tf.global_variables_initializer())
        else:
            saver.restore(sess, conf.model_path)
        for epoch in xrange(conf.max_epoch):
            train_data = data["train"]
            for img, cond in train_data:
                img, cond = prepocess_train(img, cond)
                _, m = sess.run([d_opt, model.d_loss], feed_dict={model.image:img, model.cond:cond})
                _, M = sess.run([g_opt, model.g_loss], feed_dict={model.image:img, model.cond:cond})
                counter += 1
                print "Iterate [%d]: time: %4.4f, d_loss: %.8f, g_loss: %.8f" \
                      % (counter, time.time() - start_time, m, M)
            if (epoch + 1) % conf.save_per_epoch == 0:
                save_path = saver.save(sess, conf.data_path + "/checkpoint/" + "model_%d.ckpt" % (epoch+1))
                print "Model saved in file: %s" % save_path
                test_data = data["test"]
                test_count = 0
                for img, cond in test_data:
                    test_count += 1
                    pimg, pcond = prepocess_test(img, cond)
                    gen_img = sess.run(model.gen_img, feed_dict={model.image:pimg, model.cond:pcond})
                    gen_img = gen_img.reshape(gen_img.shape[1:])
                    gen_img = (gen_img + 1.) * 127.5
                    image = np.concatenate((gen_img, cond), axis=1).astype(np.int)
                    imsave(image, conf.output_path + "/%d.jpg" % test_count)

if __name__ == "__main__":
    train()
