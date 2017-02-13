from config import Config as conf
from data import *
from model import CGAN
import tensorflow as tf

def prepocess_train(img, cond):
    img = img.reshape(1, conf.img_size, conf.img_size, conf.img_channel)
    cond = cond.reshape(1, conf.img_size, conf.img_size, conf.img_channel)
    img = img/127.5 - 1.
    cond = cond/127.5 -1.
    return img,cond

def train():
    data = load_data()
    model = CGAN()

    d_opt = tf.train.AdamOptimizer().minimize(model.d_loss)
    g_opt = tf.train.AdamOptimizer().minimize(model.g_loss)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for epoch in xrange(conf.max_epoch):
            train_data = data["train"]
            for img, cond in train_data:
                img, cond = prepocess_train(img, cond)
                _, m = sess.run([d_opt, model.d_loss], feed_dict = {model.image:img, model.cond:cond})
                _, M = sess.run([g_opt, model.g_loss], feed_dict = {model.image:img, model.cond:cond})
                print "D", m, "G", M

if __name__ == "__main__":
    train()
