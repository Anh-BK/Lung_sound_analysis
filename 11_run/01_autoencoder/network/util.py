import tensorflow as tf
import numpy as np


#================================== Utility Functions ====================================================#

def f_time_att_2d(i_tensor, att_size):
    input_shape = i_tensor.shape

    nH = input_shape[1].value
    nW = input_shape[2].value
    nC = input_shape[3].value

    # Attention para
    W_omega = tf.Variable(tf.random_normal([nH*nC, att_size],stddev=0.1)) 
    b_omega = tf.Variable(tf.random_normal([att_size],stddev=0.1)) 
    u_omega = tf.Variable(tf.random_normal([att_size],stddev=0.1)) 
 
    # Attention scheme
    v      = tf.tanh(tf.matmul(tf.reshape(i_tensor, [-1, nH*nC]), W_omega) + tf.reshape(b_omega, [1, -1]))
    vu     = tf.matmul(v, tf.reshape(u_omega, [-1, 1]))
    exps   = tf.reshape(tf.exp(vu), [-1, nW])
    o_time_att  = exps / tf.reshape(tf.reduce_sum(exps, 1), [-1, 1])

    return o_time_att





def f_sig_de_conv2d(i_tensor, ker_h, ker_w, pre_ker_num, ker_num, o_exp_h, o_exp_w, padding, stride, is_act, act_type, is_batch, mode, scope):

    # Define trainable parameters and de-conv layer
    filter_shape = [ker_h, ker_w, ker_num, pre_ker_num]  #pre_ker_num > ker_num
    b = tf.Variable(tf.constant(0.1, shape=[ker_num]), name="de_conv2d_B")
    W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="de_conv2d_W")   # this is kernel 

    #output_shape = tf.stack([50,
    output_shape = tf.stack([tf.shape(i_tensor)[0],
                             o_exp_h,
                             o_exp_w,
                             ker_num
                           ]) 

    o_de_conv2d = tf.nn.conv2d_transpose(i_tensor,
                                         W,
                                         output_shape,
                                         strides = stride,
                                         padding = padding,
                                         name    = "de_conv2d"
                                        )
    o_de_conv2d_bias = tf.nn.bias_add(o_de_conv2d, b)

    #Active function
    if(is_act==True):
        if (act_type == 'TANH'):
            o_act = tf.nn.tanh(o_de_conv2d_bias, name="TANH")
        elif ( act_type == 'RELU'): 
            o_act = tf.nn.relu(o_de_conv2d_bias, name="RELU")
        elif ( act_type == 'ELU'): 
            o_act = tf.nn.elu(o_de_conv2d_bias, name="ELU")
        else: 
            o_act = i_tensor
            print('No active function is called')
    else:
        print('No active function is called')
        o_act = i_tensor

    #Batch 2
    if(is_batch):
        o_batch = tf.contrib.layers.batch_norm(o_act,
                                               decay = 0.9,
                                               is_training = mode,
                                               zero_debias_moving_mean=True,
                                               reuse = tf.AUTO_REUSE,
                                               scope = scope
                                               )
    else:
        o_batch = o_act


    return o_batch


def f_sig_conv2d(i_tensor, ker_h, ker_w, pre_ker_num, ker_num, padding, stride, is_act, act_type, is_batch, mode, scope):

    # Define trainable parameters & Convolution layer
    filter_shape = [ker_h, ker_w, pre_ker_num, ker_num]  #pre_ker_num < ker_num
    b = tf.Variable(tf.constant(0.1, shape=[ker_num]), name="conv2d_B")
    W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="conv2d_W")   # this is kernel 

    # Convolution
    o_conv2d = tf.nn.conv2d(i_tensor,
                            W,
                            strides = stride,
                            padding = padding,
                            name    = "conv2d"
                           )
    o_conv2d_bias = tf.nn.bias_add(o_conv2d, b)

    #Active function
    if(is_act==True):
        if (act_type == 'TANH'):
            o_act = tf.nn.tanh(o_conv2d_bias, name="TANH")
        elif ( act_type == 'RELU'): 
            o_act = tf.nn.relu(o_conv2d_bias, name="RELU")
        elif ( act_type == 'ELU'): 
            o_act = tf.nn.elu(o_conv2d_bias, name="ELU")
        else:
            o_act = i_tensor
            print('No active function is called')
    else:
        print('No active function is called')
        o_act = i_tensor

    if(is_batch):
        o_batch = tf.contrib.layers.batch_norm(o_act,
                                               decay = 0.9,
                                               is_training = mode,
                                               zero_debias_moving_mean=True,
                                               reuse=tf.AUTO_REUSE,
                                               scope=scope
                                               )
    else:
        o_batch = o_act

    return o_batch


def f_sig_batch(i_tensor, mode, scope):
    o_batch = tf.contrib.layers.batch_norm(i_tensor,
                                           is_training = mode,
                                           decay = 0.9,
                                           zero_debias_moving_mean=True,
                                           reuse = tf.AUTO_REUSE,
                                           scope = scope
                                          )

    return o_batch

def f_sig_pool(i_tensor, p_type, p_padding, p_stride, p_ksize):
    if (p_type == 'MEAN'):
        o_pool = tf.nn.avg_pool(i_tensor,
                                ksize   = p_ksize,
                                strides = p_stride,
                                padding = p_padding,
                                name    = "mean_pool"
                               )
    elif (p_type == 'MAX'):
        o_pool = tf.nn.max_pool(i_tensor,
                                ksize   = p_ksize,
                                strides = p_stride,
                                padding = p_padding,
                                name    = "max_pool"
                             )
    elif (p_type == 'GLOBAL_MAX'):
        o_pool = tf.reduce_max(i_tensor,
                               axis = [1,2],
                               name = 'global_max'
                              )
    elif (p_type == 'GLOBAL_MEAN'):
        o_pool = tf.reduce_mean(i_tensor,
                                axis = [1,2],
                                name = 'global_mean_pool'
                               )
    else:
        print('No pooling is called')
        o_pool = i_tensor

    return o_pool

def f_sig_drop(i_tensor, rate, mode):
    o_drop = tf.layers.dropout(i_tensor,
                               rate     = rate,
                               training = mode,
                               name     = 'Single_Dropout'
                              )
    return o_drop

def f_sig_fully_connected(i_tensor, 
                          input_size, 
                          output_size, 
                          is_act,
                          act_func
                         ):

    #initial parameter
    W    = tf.random_normal([input_size, output_size], stddev=0.1, dtype=tf.float32)
    bias = tf.random_normal([output_size], stddev=0.1, dtype=tf.float32)
    W    = tf.Variable(W, name="fully_connect_W")
    bias = tf.Variable(bias, name="fully_connect_B")

    #Dense 
    o_dense = tf.add(tf.matmul(i_tensor, W), bias)  

    #Active function
    if(is_act == True):
        if (act_func == 'RELU'):    
            o_act_func = tf.nn.relu(o_dense, name="RELU")   
        elif (act_func == 'ELU'):
            o_act_func  = tf.nn.elu(o_dense, name="ELU")             
        elif (act_func == 'SOFTMAX'):
            o_act_func  = tf.nn.softmax(o_dense, name="SOFTMAX")             
        elif (act_func == 'TANH'):
            o_act_func  = tf.nn.tanh(o_dense, name="TANH")                 
    else:
        print('No activation function is called')
        o_act_func = o_dense

    return o_act_func


def f_sig_inct(i_tensor, pre_ker_num, ker_num, padding, stride, is_act, act_type, is_batch, mode, scope):

    # 1x1
    ker_h_01         = 1
    ker_w_01         = 1
    ker_num_01       = ker_num/8
    filter_shape_01  = [int(ker_h_01), int(ker_w_01), int(pre_ker_num), int(ker_num_01)]
    W_01             = tf.Variable(tf.truncated_normal(filter_shape_01, stddev=0.1), name="W_01")   # this is kernel 
    b_01             = tf.Variable(tf.constant(0.1, shape=[int(ker_num_01)]), name="b_01")
    conv_output_01   = tf.nn.conv2d(i_tensor,
                                    W_01,
                                    strides = stride,
                                    padding = padding,
                                    name="conv_01"
                                   )  
    conv_output_01 = tf.nn.bias_add(conv_output_01, b_01)

    # 3x3
    ker_h_02         = 3
    ker_w_02         = 3
    ker_num_02       = ker_num/2
    filter_shape_02  = [int(ker_h_02), int(ker_w_02), int(pre_ker_num), int(ker_num_02)]
    W_02             = tf.Variable(tf.truncated_normal(filter_shape_02, stddev=0.1), name="W_02")   # this is kernel 
    b_02             = tf.Variable(tf.constant(0.1, shape=[int(ker_num_02)]), name="b_02")
    conv_output_02   = tf.nn.conv2d(i_tensor,
                                    W_02,
                                    strides = stride,
                                    padding = padding,
                                    name="conv_02"
                                   ) 
    conv_output_02 = tf.nn.bias_add(conv_output_02, b_02)

    # 1x4 - learn time dim
    ker_h_03         = 1
    ker_w_03         = 4
    ker_num_03       = ker_num/4
    filter_shape_03  = [int(ker_h_03), int(ker_w_03), int(pre_ker_num), int(ker_num_03)]
    W_03             = tf.Variable(tf.truncated_normal(filter_shape_03, stddev=0.1), name="W_03")   # this is kernel 
    b_03             = tf.Variable(tf.constant(0.1, shape=[int(ker_num_03)]), name="b_03")
    conv_output_03   = tf.nn.conv2d(i_tensor,
                                    W_03,
                                    strides = stride,
                                    padding = padding,
                                    name="conv_03"
                                   )  #default: data format = NHWC
    conv_output_03 = tf.nn.bias_add(conv_output_03, b_03)


    # pooling 3x3
    pool_output_04 = tf.nn.max_pool(i_tensor,
                                    ksize   = [1,3,3,1],   
                                    strides = stride,
                                    padding = padding,
                                    name="pool_04"
                                    )
    # 1x1 - of pool
    ker_h_04         = 1
    ker_w_04         = 1
    ker_num_04       = ker_num/8
    filter_shape_04  = [int(ker_h_04), int(ker_w_04), int(pre_ker_num), int(ker_num_04)]
    W_04             = tf.Variable(tf.truncated_normal(filter_shape_04, stddev=0.1), name="W_04")   # this is kernel 
    b_04             = tf.Variable(tf.constant(0.1, shape=[int(ker_num_04)]), name="b_04")
    conv_output_04   = tf.nn.conv2d(pool_output_04,
                                    W_04,
                                    strides = stride,
                                    padding = padding,
                                    name="conv_04"
                                   )  
    conv_output_04 = tf.nn.bias_add(conv_output_04, b_04)

    # concat
    o_conv2d_bias = tf.concat((conv_output_01, conv_output_02, conv_output_03, conv_output_04),3)
    #print(conv_output.get_shape())
    #exit()

    #Active function
    if(is_act==True):
        if (act_type == 'TANH'):
            o_act = tf.nn.tanh(o_conv2d_bias, name="TANH")
        elif ( act_type == 'RELU'): 
            o_act = tf.nn.relu(o_conv2d_bias, name="RELU")
        elif ( act_type == 'ELU'): 
            o_act = tf.nn.elu(o_conv2d_bias, name="ELU")
        else: 
            o_act = i_tensor
            print('No active function is called')
    else:
        print('No active function is called')
        o_act = i_tensor


    #Batch norm
    if(is_batch):
        o_batch = tf.contrib.layers.batch_norm(o_act,
                                               decay = 0.9,
                                               is_training = mode,
                                               zero_debias_moving_mean=True,
                                               reuse = tf.AUTO_REUSE,
                                               scope = scope
                                               )
    else:
        o_batch = o_act

    return o_batch
#===================================================================================
