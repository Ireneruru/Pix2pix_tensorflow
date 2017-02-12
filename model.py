import tensorflow as tf
import numpy as np
from config import Config
from glob import glob
from data import *


class CGAN(object):
    def __init__(self):
        self.image = tf.placeholder()
        self.cond = tf.placeholder()
        self.noise = tf.placeholer()
        self.gen_img = self.generator(self.noise, self.cond)
        pos = self.D(self.image, self.cond)
        pos_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=pos, labels=tf.ones.like(pos)))
        neg = self.G(self.gen_img, self.cond)
        neg_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=neg, labels=tf.zeros.like(neg)))
        self.d_loss = pos_loss + neg_loss
        self.g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=neg, labels=tf.ones.like(neg))) + \
                      Config.L1_lambda * tf.reduce_mean(tf.abs(self.image - self.gen_img))

    def D(img, cond):
        dim = len (img.shape())
        with tf.variable.scope("disc"):
            concat = tf.concat(dim-1, [img, cond])
            # conv2d layers

            # linear regression layer

            #return result



    def G(noise, cond):
        with tf.variable_scope("gen"):
            pass





