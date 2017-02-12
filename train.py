from config import Config
from data import *
from model import CGAN
import tensorflow as tf

def train():
    data = load_data()
    model = CGAN()

    d_opt = tf.train.AdadeltaOptimizer.minimize(model.d_loss)
    g_opt = tf.train.AdadeltaOptimizer.minimize(model.g_loss)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for epoch in xrange(Config.max_epoch):
            train_data = data["train"]
            for img, cond in train_data:
                sess.run(d_opt, feed_dict = {model.image:img, model.cond:cond})
                sess.run(g_opt, feed_dict = {model.image:img, model.cond:cond})


if __name__ == "__main__":
    train()
