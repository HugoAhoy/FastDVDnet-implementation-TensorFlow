# rewrite use sepconv and reuse module
import os
import cv2
import numpy as np
import tensorflow as tf
from pixelShuffle import PS_tf as PS
from utils import open_sequence, variable_to_cv2_image, open_image
from config import *
import time

# TENSORFLOW USING NHWC
def SepConv(input, in_ch, out_ch, kernel_size=(3,3), stride=(1,1), channel_multiplier=1, name=None, reuse=None):
    assert input.shape[-1] == in_ch
    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, kernel_size)
    else:
        assert isinstance(kernel_size, tuple)
        assert len(kernel_size) == 2
        assert isinstance(kernel_size[0],int) and isinstance(kernel_size[-1],int)
        assert kernel_size[0] == kernel_size[1]

    if isinstance(stride, int):
        stride = (1,stride, stride,1)
    else:
        assert isinstance(stride, tuple)
        assert len(stride) == 2
        assert isinstance(stride[0],int) and isinstance(stride[-1],int)
        stride = (1,stride[0],stride[1],1)

    with tf.variable_scope(name, default_name="SepConv",reuse=reuse) as sep_scope:
        # paddings1 = tf.constant([[0,0],[kernel_size[0]//2, kernel_size[0]//2,],[kernel_size[1]//2, kernel_size[1]//2,], [0, 0]])
        '''
        tf.Varible() can't share var, replace it by tf.get_variable()
        '''
        # depthwisekernel = tf.Variable(tf.random_normal(shape=[kernel_size[0],kernel_size[1], in_ch, channel_multiplier],mean=0,stddev=1),name='depthwise_kernel')
        # pointwisekernel = tf.Variable(tf.random_normal(shape=[1,1, channel_multiplier*in_ch, out_ch],mean=0,stddev=1),name='pointwise_kernel')
        depthwisekernel = tf.get_variable(name='depthwise_kernel',shape=[kernel_size[0],kernel_size[1], in_ch, channel_multiplier],dtype=tf.float32)
        pointwisekernel = tf.get_variable(name='pointwise_kernel',shape=[1,1, channel_multiplier*in_ch, out_ch],dtype=tf.float32)

        if stride[1] == 1:
            out = tf.nn.separable_conv2d(input, depthwisekernel, pointwisekernel, stride, padding='SAME', name="sepconv")
            return out
        else:
            if kernel_size[0] == 3:
                pad1 = tf.pad(input, padding_3_3)
            elif kernel_size[0] == 5:
                pad1 = tf.pad(input, padding_5_5)
            else:
                raise ValueError("kernel size {} is not supported.".format(kernel_size[0]))\
            
            # pad1 = tf.pad(input, padding1)
            out = tf.nn.separable_conv2d(pad1, depthwisekernel, pointwisekernel, stride, padding='VALID', name="sepconv")
            return out

def CvBlock(input, in_ch, out_ch,training, name=None, reuse=None):
    with tf.variable_scope(name, default_name="CvBlock",reuse=reuse) as cb_scope:
        conv1 = tf.layers.conv2d(input, out_ch, 3, padding='SAME', use_bias=False)
        bn1 = tf.layers.batch_normalization(conv1, momentum=0.1, epsilon=1e-5, training = training)
        relu1 = tf.nn.relu(bn1)
        conv2 = tf.layers.conv2d(relu1, out_ch, 3, padding='SAME', use_bias=False)
        bn2 = tf.layers.batch_normalization(conv2, momentum=0.1, epsilon=1e-5, training = training)
        relu2 = tf.nn.relu(bn2)
        return relu2


def InputCvBlock_3(input, in_ch, out_ch,training, num_in_frame=3,name=None, reuse=None):
    interm_ch=30

    with tf.variable_scope(name, default_name="InputCvBlock_3",reuse=reuse) as input_scope:
        out = SepConv(input, in_ch, interm_ch*num_in_frame, kernel_size=3, name="sep1")
        bn1 = tf.layers.batch_normalization(out, momentum=0.1, epsilon=1e-5, training = training, name="bn1")
        relu1 = tf.nn.relu(bn1, name="relu1")
        conv2 = tf.layers.conv2d(relu1, out_ch, 3, padding='SAME', use_bias=False, name="conv2")
        bn2 = tf.layers.batch_normalization(conv2, momentum=0.1, epsilon=1e-5, training = training,name="bn2")
        relu2 = tf.nn.relu(bn2, name="relu2")
        return relu2


paddings1 = tf.constant([[0,0],[1, 1,],[1, 1,], [0, 0]])

def DownBlock(input, in_ch, out_ch, training, name=None, reuse=None):
    # 指定padding
    with tf.variable_scope(name, default_name="DownBlock",reuse=reuse) as db_scope:
        pad1 = tf.pad(input, paddings1)
        conv1 = tf.layers.conv2d(pad1, out_ch, 3, padding='VALID', strides=2, use_bias=False,name="conv1")
        bn1 = tf.layers.batch_normalization(conv1, momentum=0.1, epsilon=1e-5, training = training,name="bn1")
        relu1 = tf.nn.relu(bn1,name="relu1")
        output = CvBlock(relu1,out_ch, out_ch, training = training, name="CvBlock1")
        return output


def UpBlock(input, in_ch, out_ch, training, name=None, reuse=None):
    with tf.variable_scope(name, default_name="DownBlock",reuse=reuse) as db_scope:
        out1 = CvBlock(input, in_ch,in_ch, training = training, name="CvBlock1")
        conv1 = tf.layers.conv2d(out1, out_ch*4, 3, padding='SAME', use_bias=False,name="conv1")
        output = PS(conv1, 2)
        return output


# def OutputCvBlock(out_ch, input, reuse=False):
def OutputCvBlock(input, in_ch, out_ch, training, name=None, reuse=None):
    with tf.variable_scope(name, default_name="DownBlock",reuse=reuse) as db_scope:
        conv1 = tf.layers.conv2d(input, in_ch, 3, padding='SAME', use_bias=False,name="conv1")
        bn1 = tf.layers.batch_normalization(conv1, momentum=0.1, epsilon=1e-5, training = training,name="bn1")
        relu1 = tf.nn.relu(bn1,name="relu1")
        conv2 = tf.layers.conv2d(relu1, out_ch, 3, padding='SAME', use_bias=False,name="conv2")
        return conv2


def DenBlock(in0, in1, in2, noise_map, training, outName = None, name=None, reuse=None):

    with tf.variable_scope(name, default_name="DenBlock",reuse=reuse) as den_scope:
        input = tf.concat([in0, noise_map,in1, noise_map,in2, noise_map],3)
        chs_lyr0 = 32
        chs_lyr1 = 64
        chs_lyr2 = 128

        # input0 = tf.concat([in0, noise_map], 3)
        # input1 = tf.concat([in1, noise_map], 3)
        # input2 = tf.concat([in2, noise_map], 3)

        inc = InputCvBlock_3( input,3*(3+1),chs_lyr0, training = training, name="InputBlock1")
        downc0 = DownBlock(inc, chs_lyr0, chs_lyr1, training = training,name="DownBlock1")
        downc1 = DownBlock(downc0, chs_lyr1,chs_lyr2, training = training,name="DownBlock2")

        # self.upc2 = UpBlock(in_ch=self.chs_lyr2, out_ch=self.chs_lyr1)
        # self.upc1 = UpBlock(in_ch=self.chs_lyr1, out_ch=self.chs_lyr0)
        # 以上为torch源码中的upconv,upc1的out_ch=chs_lyr0
        upc2 = UpBlock(downc1, chs_lyr2, chs_lyr1, training = training,name="UpBlock2")
        # upc1 = UpBlock(chs_lyr1, downc0+upc2)
        upc1 = UpBlock(downc0+upc2, chs_lyr1, chs_lyr0, training = training,name="UpBlock1")

        # outc = OutputCvBlock(3, upc1)
        # x1 = self.upc1(x1+x2)
        # x = self.outc(x0+x1)
        # 以上为torch源码中的upconv,upc1的out_ch=chs_lyr0
        outc = OutputCvBlock(upc1+inc, chs_lyr0, 3, training = training, name="OutputBlock1")

        # output = in1 - outc
        output = tf.math.subtract(in1, outc, name = outName)

        return output


def fastDVDnet(input, noise_map, training,name=None,reuse=None):

    with tf.variable_scope(name, default_name="FastDvdNet",reuse=reuse) as fdn_scope:
        num_input_frames = 5

        (x0, x1, x2, x3, x4) = tuple(input[:, :, :, 3*m:3*(m+1)] for m in range(num_input_frames))

        x20 = DenBlock(x0, x1, x2, noise_map, training = training,name="DenBlock1",reuse=tf.AUTO_REUSE)
        x21 = DenBlock(x1, x2, x3, noise_map, training = training,name="DenBlock1",reuse=tf.AUTO_REUSE)
        x22 = DenBlock(x2, x3, x4, noise_map, training = training,name="DenBlock1",reuse=tf.AUTO_REUSE)

        x = DenBlock(x20,x21,x22, noise_map, training = training, outName = "out",name="DenBlock2")

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
        print("build graph successfully!")

        return output

    def test_run(self):
        seqn = np.ones([1, self.h, self.w, self.c])
        noise_map = np.ones([1, self.h, self.w, 1])


        feed_dict = {
            self.input_image: seqn,
            self.input_noise_map: noise_map,
        }

        output = self.sess.run(self.output, feed_dict=feed_dict)
        print("pipeline runs successfully!")

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
        output_graph = "./model/pb/fastdvdnet_8-25_{}x{}.pb".format(self.h,self.w)
        # self.restoreModel()
        graph_def = tf.compat.v1.get_default_graph().as_graph_def()
        output_graph_def = tf.graph_util.convert_variables_to_constants(
            self.sess,
            graph_def,
            ["FastDvdNet/DenBlock2/out"] #需要保存节点的名字
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
    # h,w,c = 360,640,15
    # h,w,c = 540,960,15
    h,w,c = 720,1280,15
    with tf.Session() as sess:
        test_model = FastDVDNet(sess, h, w, c)
        sess.run(tf.global_variables_initializer())
        # output = test_model.test()
        # img = variable_to_cv2_image(output)
        # cv2.imwrite('./testResult.png', img)
        test_model.savePB()
        for variable_name in tf.global_variables():
            print(variable_name)
        test_model.test_run()
