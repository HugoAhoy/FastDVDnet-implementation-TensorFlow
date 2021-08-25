import tensorflow as tf
import numpy as np

def _phase_shift(I, r):
    #print("I:",I,"I.get_shape():",I.get_shape().as_list())
    bsize, h, w, c = I.get_shape().as_list()
    bsize = tf.shape(I)[0] # Handling Dimension(None) type for undefined batch dim
    X = tf.reshape(I, (bsize, h, w, r, r))
    #X = tf.transpose(X, (0, 1, 2, 4, 3))  # bsize, a, b, 1, 1
    X = tf.split(X, h, 1)  # a, [bsize, b, r, r]
    X = tf.concat([tf.squeeze(x, axis=1) for x in X], 2)  # bsize, b, a*r, r
    X = tf.split(X, w, 1)  # b, [bsize, a*r, r]
    X = tf.concat([tf.squeeze(x, axis=1) for x in X], 2)  # bsize, a*r, b*r
   
    return tf.reshape(X, (bsize, h*r, w*r, 1))

def PS(X, r):
    # print("Input X shape:",X.get_shape(),"scale:",r)
    bsize, w, h, c = X.get_shape().as_list()

    inv = int(c/(r**2))

    Xc = tf.split(X, inv, 3) 

    X = tf.concat([_phase_shift(x, r) for x in Xc], 3)
    return X

def shuffle_channel_tf(x, ch=1):
    num_ch = int(x.shape[ch])
    out_num_ch = num_ch // 4
    index = []
    for i in range(out_num_ch):
        index += list(range(i, num_ch, out_num_ch))
    # index = list(range(0, num_ch, fold)) + list(range(1, num_ch, fold))
    inv_index = np.zeros(num_ch, dtype=np.int32)
    for i in range(num_ch):
        inv_index[index[i]] = i
    return tf.gather(x, inv_index.tolist(), axis=ch)


def PS_tf(X, r=2):
    assert r == 2
    X = shuffle_channel_tf(X, ch=3)
    return tf.depth_to_space(X, r)

if __name__== "__main__":
    import time
    w, h, c = 270,480, 12
    a = np.arange(1, 1+w*h*c*0.1, 0.1).astype(np.float32).reshape(1,w,h,c)
    a1 = np.arange(1, 1+w*h*c*0.1, 0.1).astype(np.float32).reshape(1,w,h,c)
    a = tf.convert_to_tensor(a)
    a1 = tf.convert_to_tensor(a1)
    b = PS(a, 2)
    c = PS_tf(a1, 2)
    with tf.Session() as sess:
        # sess.run(c)
        # sess.run(b)
        tb = 0
        tc = 0
        for i in range(1000):
            t3 = time.time()
            sess.run(c)
            t4 = time.time()

            t1 = time.time()
            sess.run(b)
            t2 = time.time()

            tb += t2-t1
            tc += t4-t3

        print(tb)
        print(tc)
        print(tb/tc)
