import os
import cv2
import numpy as np
import tensorflow as tf
from pixelShuffle import PS_tf as PS
from utils import open_sequence, variable_to_cv2_image, open_image
from config import *
import time

# TENSORFLOW USING NHWC

def CvBlock(out_ch,input, training, reuse=False):

    #with tf.variable_scope(name, reuse=reuse):
    conv1 = tf.layers.conv2d(input, out_ch, 3, padding='SAME', use_bias=False)
    bn1 = tf.layers.batch_normalization(conv1, momentum=0.1, epsilon=1e-5, training = training)
    relu1 = tf.nn.relu(bn1)
    conv2 = tf.layers.conv2d(relu1, out_ch, 3, padding='SAME', use_bias=False)
    bn2 = tf.layers.batch_normalization(conv2, momentum=0.1, epsilon=1e-5, training = training)
    relu2 = tf.nn.relu(bn2)
    return relu2


def InputCvBlock_3(out_ch,
                frame1, frame2, frame3, training, reuse=False):
    interm_ch=30

    #with tf.variable_scope(name, reuse=reuse):
    conv1_1 = tf.layers.conv2d(frame1, interm_ch, 3, padding='SAME', use_bias=False)
    conv1_2 = tf.layers.conv2d(frame2, interm_ch, 3, padding='SAME', use_bias=False)
    conv1_3 = tf.layers.conv2d(frame3, interm_ch, 3, padding='SAME', use_bias=False)

    cat = tf.concat([conv1_1, conv1_2, conv1_3], 3)
    bn1 = tf.layers.batch_normalization(cat, momentum=0.1, epsilon=1e-5, training = training)
    relu1 = tf.nn.relu(bn1)
    # conv2 = tf.layers.conv2d(relu1, out_ch, 3, padding='SAME', activation=tf.nn.relu, use_bias=False)
    # 之前多写了一个activation
    conv2 = tf.layers.conv2d(relu1, out_ch, 3, padding='SAME', use_bias=False)
    bn2 = tf.layers.batch_normalization(conv2, momentum=0.1, epsilon=1e-5, training = training)
    relu2 = tf.nn.relu(bn2)

    return relu2


def InputCvBlock_5(out_ch,
                frame1, frame2, frame3, frame4, frame5, reuse=False):
    interm_ch=30

    #with tf.variable_scope(name, reuse=reuse):
    conv1_1 = tf.layers.conv2d(frame1, interm_ch, 3, padding='SAME', use_bias=False)
    conv1_2 = tf.layers.conv2d(frame2, interm_ch, 3, padding='SAME', use_bias=False)
    conv1_3 = tf.layers.conv2d(frame3, interm_ch, 3, padding='SAME', use_bias=False)
    conv1_4 = tf.layers.conv2d(frame4, interm_ch, 3, padding='SAME', use_bias=False)
    conv1_5 = tf.layers.conv2d(frame5, interm_ch, 3, padding='SAME', use_bias=False)
    cat = tf.concat(3,[conv1_1, conv1_2, conv1_3, conv1_4, conv1_5])
    bn1 = tf.layers.batch_normalization(cat, momentum=0.1, epsilon=1e-5)
    relu1 = tf.nn.relu(bn1)
    conv2 = tf.layers.conv2d(relu1, out_ch, 3, padding='SAME', activation=tf.nn.relu, use_bias=False)
    bn2 = tf.layers.batch_normalization(conv2, momentum=0.1, epsilon=1e-5)
    relu2 = tf.nn.relu(bn2)

    return relu2


def DownBlock(out_ch, input, training, reuse=False):
    # 指定padding
    paddings1 = tf.constant([[0,0],[1, 1,],[1, 1,], [0, 0]])
    pad1 = tf.pad(input, paddings1)
    # conv1 = tf.layers.conv2d(input, out_ch, 3, padding='SAME', strides=2, use_bias=False)
    conv1 = tf.layers.conv2d(pad1, out_ch, 3, padding='VALID', strides=2, use_bias=False)
    bn1 = tf.layers.batch_normalization(conv1, momentum=0.1, epsilon=1e-5, training = training)
    relu1 = tf.nn.relu(bn1)
    output = CvBlock(out_ch, relu1, training = training)
    return output


# def UpBlock(out_ch, input, reuse=False):
def UpBlock(in_ch, out_ch, input, training, reuse=False):
    # out1 = CvBlock(out_ch, input)
    out1 = CvBlock(in_ch, input, training = training)
    conv1 = tf.layers.conv2d(out1, out_ch*4, 3, padding='SAME', use_bias=False)
    # output = tf.nn.depth_to_space(conv1, 2) # probably different
    output = PS(conv1, 2)
    return output


# def OutputCvBlock(out_ch, input, reuse=False):
def OutputCvBlock(in_ch, out_ch, input, training, reuse=False):
    # conv1 = tf.layers.conv2d(input, out_ch, 3, padding='SAME', use_bias=False)
    conv1 = tf.layers.conv2d(input, in_ch, 3, padding='SAME', use_bias=False)
    bn1 = tf.layers.batch_normalization(conv1, momentum=0.1, epsilon=1e-5, training = training)
    relu1 = tf.nn.relu(bn1)
    conv2 = tf.layers.conv2d(relu1, out_ch, 3, padding='SAME', use_bias=False)
    return conv2


def DenBlock(in0, in1, in2, noise_map, training, reuse=False, outName = None):

    assert input is not None
    chs_lyr0 = 32
    chs_lyr1 = 64
    chs_lyr2 = 128

    input0 = tf.concat([in0, noise_map], 3)
    input1 = tf.concat([in1, noise_map], 3)
    input2 = tf.concat([in2, noise_map], 3)

    inc = InputCvBlock_3(chs_lyr0, input0, input1, input2, training = training)
    downc0 = DownBlock(chs_lyr1, inc, training = training)
    downc1 = DownBlock(chs_lyr2, downc0, training = training)

    # self.upc2 = UpBlock(in_ch=self.chs_lyr2, out_ch=self.chs_lyr1)
    # self.upc1 = UpBlock(in_ch=self.chs_lyr1, out_ch=self.chs_lyr0)
    # 以上为torch源码中的upconv,upc1的out_ch=chs_lyr0
    upc2 = UpBlock(chs_lyr2, chs_lyr1, downc1, training = training)
    # upc1 = UpBlock(chs_lyr1, downc0+upc2)
    upc1 = UpBlock(chs_lyr1, chs_lyr0, downc0+upc2, training = training)

    # outc = OutputCvBlock(3, upc1)
    # x1 = self.upc1(x1+x2)
    # x = self.outc(x0+x1)
    # 以上为torch源码中的upconv,upc1的out_ch=chs_lyr0
    outc = OutputCvBlock(chs_lyr0, 3, upc1+inc, training = training)

    # output = in1 - outc
    output = tf.math.subtract(in1, outc, name = outName)

    return output


def fastDVDnet(input, noise_map, training):

    num_input_frames = 5

    (x0, x1, x2, x3, x4) = tuple(input[:, :, :, 3*m:3*m+3] for m in range(num_input_frames))

    x20 = DenBlock(x0, x1, x2, noise_map, training = training)
    x21 = DenBlock(x1, x2, x3, noise_map, training = training)
    x22 = DenBlock(x2, x3, x4, noise_map, training = training)

    x = DenBlock(x20,x21,x22, noise_map, training = training, outName = "out")

    return x


class FastDVDNet():
    def __init__(self, sess, h=540, w=960, c=15):
        self.sess = sess
        self.dtype=tf.float32
        self.output_dir = './'
        self.checkpoint_dir = os.path.join(self.output_dir, "checkpoint")
        self.checkpoint_prefix = "model"
        self.saver_name = "checkpoint"
        self.summary_dir = os.path.join(self.output_dir, "summary")
        self.h = h
        self.w = w
        self.c = c

        self.input_image = tf.placeholder(self.dtype, [None, h, w, c])
        self.input_noise_map = tf.placeholder(self.dtype, [None, h, w, 1])
        self.output = self._build_test()


    def _build_test(self):
        output = fastDVDnet(self.input_image, self.input_noise_map, False)
        self.saver = tf.train.Saver(max_to_keep=10, name= self.saver_name)

        return output

    def test(self):
        seq, _, _ = open_sequence(TEST_SEQUENCE_DIR, \
                                  GRAY, \
                                  expand_if_needed=False, \
                                  max_num_fr=MAX_NUM_FR_PER_SEQ)

        # Add noise
    	# Normalize noises ot [0, 1]
        noise_std = NOISE_SIGMA/255.
        # print(noise_std)
        noise = np.random.normal(loc=0, scale=noise_std, size=seq.shape)
        seqn = seq + noise
        noise_map = np.ones([1, 540, 960, 1]) * noise_std

        feed_dict = {
            self.input_image: seqn,
            self.input_noise_map: noise_map,
        }
        ckpt_state = tf.train.get_checkpoint_state(self.checkpoint_dir)
        if ckpt_state:
            self.saver.restore(self.sess, ckpt_state.model_checkpoint_path)
            print("model restored")
        else:
            print("no checkpoint model")
            self.sess.run(tf.global_variables_initializer())
        t1 = time.time()
        output = self.sess.run(self.output, feed_dict=feed_dict)
        t2 = time.time()
        print("cost:", t2 - t1)
        return output
    
    def saveModel(self):
        self.saver.save(\
            self.sess, os.path.join( \
                                self.checkpoint_dir, \
                                self.checkpoint_prefix))
    def restoreModel(self):
        ckpt_state = tf.train.get_checkpoint_state(self.checkpoint_dir)
        if ckpt_state:
            self.saver.restore(self.sess, ckpt_state.model_checkpoint_path)
            print("model restored")
    
    def savePB(self):
        output_node_names = "out"
        output_graph = "./model/pb/fastdvdnet_{}x{}.pb".format(self.h,self.w)
        self.restoreModel()
        graph_def = tf.compat.v1.get_default_graph().as_graph_def()
        output_graph_def = tf.graph_util.convert_variables_to_constants(
            self.sess,
            graph_def,
            ["out"] #需要保存节点的名字
        )
        print("get_output_graph_def")
        with tf.gfile.GFile(output_graph, "wb") as f:  # 保存模型
            f.write(output_graph_def.SerializeToString())  # 序列化输出
            print("done")
            print("%d ops in the final graph." % len(output_graph_def.node))

if __name__ == '__main__':
    # os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    # config = tf.ConfigProto()
    # config.gpu_options.per_process_gpu_memory_fraction = 0.5
    # with tf.Session(config=config) as sess:
    h,w,c = 360,640,15
    # h,w,c = 540,960,15
    with tf.Session() as sess:
        test_model = FastDVDNet(sess, h, w, c)
        # output = test_model.test()
        # img = variable_to_cv2_image(output)
        # cv2.imwrite('./testResult.png', img)
        test_model.savePB()
