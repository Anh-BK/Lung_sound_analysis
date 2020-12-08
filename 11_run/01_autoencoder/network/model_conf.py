import tensorflow as tf
from model_para import *
from util       import *

class model_conf(object):
    def __init__(self):
        
        # ================ Setting
        #Get para
        self.para          = model_para()

        #Input Fed
        self.i_image       = tf.placeholder(shape = [None, self.para.i_nF, self.para.i_nT, self.para.i_nC],  
                                            dtype = tf.float32, 
                                            name  = "input_real_image"
                                           )

        self.mode          = tf.placeholder(dtype = tf.bool, 
                                            name  = "running_mode"
                                           )

        #Output Fed
        self.i_exp_class   = tf.placeholder(shape = [None, self.para.n_class], 
                                            dtype = tf.float32, 
                                            name  = "expected_classes"
                                           )

        #============== Network construction
        with tf.device('/gpu:0'), tf.variable_scope("ENC_para"):  
            #self.latent, self.pred_score = self.encoder(i_image = self.i_image, 
            self.latent = self.encoder(i_image = self.i_image, 
                                       mode    = self.mode 
                                      )

        with tf.device('/gpu:0'), tf.variable_scope("DEC_para"):
            self.fake_image = self.decoder(i_latent = self.latent,  
                                           mode     = self.mode
                                          )
            self.fake_image = tf.reshape(self.fake_image, [-1, self.para.i_nF, self.para.i_nT, self.para.i_nC])

            #For extraction
            #self.pos_prob   = tf.nn.softmax(self.pred_score)


        #============ Loss function
        with tf.device('/gpu:0'), tf.variable_scope("Loss_def"):  
            # l2 loss  
            self.loss_nor = self.para.l2_lamda * sum(tf.nn.l2_loss(tf_var) for tf_var in tf.trainable_variables())

            # class loss - entropy
            #self.loss_class = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.i_exp_class, logits=self.pred_score))

            ## class loss - KL
            #dummy = tf.constant(0.00001) # to avoid dividing by 0 with KL divergence
            #p = self.i_exp_class + dummy
            #q = self.pos_prob + dummy
            #self.loss_class = tf.reduce_sum(p * tf.log(p/q))

            # Loss compares real and face image
            self.loss_cmp  = tf.reduce_mean(tf.losses.mean_squared_error(self.i_image, self.fake_image), name='loss')

            # Final loss
            #self.losses = self.loss_nor + self.loss_class + self.loss_cmp
            self.losses = self.loss_nor + self.loss_cmp

        #===========  Calculate Accuracy  
        #with tf.device('/gpu:0'), tf.name_scope("Accuracy") as scope:
        #    self.cor_pred = tf.equal(tf.argmax(self.pred_score,1), tf.argmax(self.i_exp_class,1))
        #    self.accuracy = tf.reduce_mean(tf.cast(self.cor_pred,"float", name="accuracy_real" ))

    #=============== Function Def
    def decoder (self, i_latent, mode):
        
        #i_latent: vector  266-dim

        #========= Block 01/ 
        #Fully_connected layer
        with tf.device('/gpu:0'), tf.variable_scope("DEC_bl01_fc"):  
            o_bl01_fc = f_sig_fully_connected(i_tensor    = i_latent,        
                                              input_size  = self.para.dec_bl01_fc_input_size,
                                              output_size = self.para.dec_bl01_fc_output_size,
                                              is_act      = self.para.dec_bl01_fc_is_act,
                                              act_func    = self.para.dec_bl01_fc_act_func
                                             )
        #Reshape
        with tf.device('/gpu:0'), tf.variable_scope("DEC_bl01_reshape"):  
            o_bl01_reshape = tf.reshape(tensor = o_bl01_fc, 
                                        shape  = [-1, self.para.dec_bl01_nF, self.para.dec_bl01_nT, self.para.dec_bl01_nC],
                                        name   = "reshape"
                                       )

        #========= Block 02/
        #De-conv2d 
        with tf.device('/gpu:0'), tf.variable_scope("DEC_bl02_dconv2d") as scope:  
            o_bl02_deconv2d = f_sig_de_conv2d(i_tensor    = o_bl01_reshape,
                                              ker_h       = self.para.dec_bl02_ker_h,
                                              ker_w       = self.para.dec_bl02_ker_w,
                                              pre_ker_num = self.para.dec_bl02_pre_ker_num,
                                              ker_num     = self.para.dec_bl02_ker_num,
                                              o_exp_h     = self.para.dec_bl02_o_exp_h,
                                              o_exp_w     = self.para.dec_bl02_o_exp_w,
                                              padding     = self.para.dec_bl02_padding,
                                              stride      = self.para.dec_bl02_stride,
                                              is_act      = self.para.dec_bl02_is_act,
                                              act_type    = self.para.dec_bl02_act_type,
                                              is_batch    = self.para.dec_bl02_is_batch,
                                              mode        = self.mode,
                                              scope       = scope
                                             ) 
            
        print(o_bl02_deconv2d.get_shape())    
        #========= Block 03/
        #De-conv2d 
        with tf.device('/gpu:0'), tf.variable_scope("DEC_bl03_dconv2d") as scope:  
            o_bl03_deconv2d = f_sig_de_conv2d(i_tensor    = o_bl02_deconv2d,
                                              ker_h       = self.para.dec_bl03_ker_h,
                                              ker_w       = self.para.dec_bl03_ker_w,
                                              pre_ker_num = self.para.dec_bl03_pre_ker_num,
                                              ker_num     = self.para.dec_bl03_ker_num,
                                              o_exp_h     = self.para.dec_bl03_o_exp_h,
                                              o_exp_w     = self.para.dec_bl03_o_exp_w,
                                              padding     = self.para.dec_bl03_padding,
                                              stride      = self.para.dec_bl03_stride,
                                              is_act      = self.para.dec_bl03_is_act,
                                              act_type    = self.para.dec_bl03_act_type,
                                              is_batch    = self.para.dec_bl02_is_batch,
                                              mode        = self.mode,
                                              scope       = scope
                                             ) 
            print(o_bl03_deconv2d.get_shape())    
        #========= Block 04/
        #De-conv2d 
        with tf.device('/gpu:0'), tf.variable_scope("DEC_bl04_dconv2d") as scope:  
            o_bl04_deconv2d = f_sig_de_conv2d(i_tensor    = o_bl03_deconv2d,
                                              ker_h       = self.para.dec_bl04_ker_h,
                                              ker_w       = self.para.dec_bl04_ker_w,
                                              pre_ker_num = self.para.dec_bl04_pre_ker_num,
                                              ker_num     = self.para.dec_bl04_ker_num,
                                              o_exp_h     = self.para.dec_bl04_o_exp_h,
                                              o_exp_w     = self.para.dec_bl04_o_exp_w,
                                              padding     = self.para.dec_bl04_padding,
                                              stride      = self.para.dec_bl04_stride,
                                              is_act      = self.para.dec_bl04_is_act,
                                              act_type    = self.para.dec_bl04_act_type,
                                              is_batch    = self.para.dec_bl02_is_batch,
                                              mode        = self.mode,
                                              scope       = scope
                                             ) 
            print(o_bl04_deconv2d.get_shape())    

        #========= Block 05/
        #De-conv2d 
        with tf.device('/gpu:0'), tf.variable_scope("DEC_bl05_dconv2d") as scope:  
            o_bl05_deconv2d = f_sig_de_conv2d(i_tensor    = o_bl04_deconv2d,
                                              ker_h       = self.para.dec_bl05_ker_h,
                                              ker_w       = self.para.dec_bl05_ker_w,
                                              pre_ker_num = self.para.dec_bl05_pre_ker_num,
                                              ker_num     = self.para.dec_bl05_ker_num,
                                              o_exp_h     = self.para.dec_bl05_o_exp_h,
                                              o_exp_w     = self.para.dec_bl05_o_exp_w,
                                              padding     = self.para.dec_bl05_padding,
                                              stride      = self.para.dec_bl05_stride,
                                              is_act      = self.para.dec_bl05_is_act,
                                              act_type    = self.para.dec_bl05_act_type,
                                              is_batch    = self.para.dec_bl02_is_batch,
                                              mode        = self.mode,
                                              scope       = scope
                                             ) 
            print(o_bl05_deconv2d.get_shape())    


        return o_bl05_deconv2d

    #----------------------------------------- ENCODER CONFIGURATION -----------------------------------#
    def encoder (self, i_image, mode):
        
        #input : i_image:  nBxnFxnTxnC
        #return: latent, predict

        #========= Block 01
        #Batch
        with tf.device('/gpu:0'), tf.variable_scope("enc_bl01_batch") as scope:  
            o_bl01_batch = f_sig_batch(i_tensor = i_image, 
                                       mode     = mode,
                                       scope    = scope
                                      )
        #Conv01 
        with tf.device('/gpu:0'), tf.variable_scope("ENC_bl01_conv2d") as scope:  
            o_bl01_conv2d = f_sig_inct(i_tensor    = o_bl01_batch,
                                         #ker_h       = self.para.enc_bl01_ker_h,
                                         #ker_w       = self.para.enc_bl01_ker_w,
                                         pre_ker_num = self.para.enc_bl01_pre_ker_num,
                                         ker_num     = self.para.enc_bl01_ker_num,
                                         padding     = self.para.enc_bl01_padding,
                                         stride      = self.para.enc_bl01_stride,
                                         is_act      = self.para.enc_bl01_is_act,
                                         act_type    = self.para.enc_bl01_act_type,
                                         is_batch    = self.para.enc_bl01_is_batch,
                                         mode        = mode,
                                         scope       = scope
                                        )
        #Pooling 01
        with tf.device('/gpu:0'), tf.variable_scope("ENC_bl01_pool"):  
            o_bl01_pool = f_sig_pool(i_tensor  = o_bl01_conv2d,
                                     p_type    = self.para.enc_bl01_p_type,
                                     p_padding = self.para.enc_bl01_p_padding,
                                     p_stride  = self.para.enc_bl01_p_stride,
                                     p_ksize   = self.para.enc_bl01_p_ksize
                                    ) 
       
        with tf.device('/gpu:0'), tf.variable_scope("ENC_bl01_drop"):  
            o_bl01_drop = f_sig_drop(i_tensor = o_bl01_pool,
                                     rate     = self.para.enc_bl01_drop_prob,
                                     mode     = mode
                                    )
            o_bl01_output = o_bl01_drop   

        #========= Block 02
        #Conv02 
        with tf.device('/gpu:0'), tf.variable_scope("ENC_bl02_conv2d") as scope:  
            o_bl02_conv2d = f_sig_inct(i_tensor  = o_bl01_output,
                                         #ker_h       = self.para.enc_bl02_ker_h,
                                         #ker_w       = self.para.enc_bl02_ker_w,
                                         pre_ker_num = self.para.enc_bl02_pre_ker_num,
                                         ker_num     = self.para.enc_bl02_ker_num,
                                         padding     = self.para.enc_bl02_padding,
                                         stride      = self.para.enc_bl02_stride,
                                         is_act      = self.para.enc_bl02_is_act,
                                         act_type    = self.para.enc_bl02_act_type,
                                         is_batch    = self.para.enc_bl02_is_batch,
                                         mode        = mode,
                                         scope       = scope
                                        )
        #Pooling 02
        with tf.device('/gpu:0'), tf.variable_scope("ENC_bl02_pool"):  
            o_bl02_pool = f_sig_pool(i_tensor  = o_bl02_conv2d,
                                     p_type    = self.para.enc_bl02_p_type,
                                     p_padding = self.para.enc_bl02_p_padding,
                                     p_stride  = self.para.enc_bl02_p_stride,
                                     p_ksize   = self.para.enc_bl02_p_ksize
                                    ) 

        with tf.device('/gpu:0'), tf.variable_scope("ENC_bl02_drop"):  
            o_bl02_drop = f_sig_drop(i_tensor = o_bl02_pool,
                                     rate     = self.para.enc_bl02_drop_prob,
                                     mode     = mode
                                    )
            o_bl02_output = o_bl02_drop   

        #========= Block 03
        #Conv03 
        with tf.device('/gpu:0'), tf.variable_scope("ENC_bl03_conv2d") as scope:  
            o_bl03_conv2d = f_sig_inct(i_tensor    = o_bl02_output,
                                         #ker_h       = self.para.enc_bl03_ker_h,
                                         #ker_w       = self.para.enc_bl03_ker_w,
                                         pre_ker_num = self.para.enc_bl03_pre_ker_num,
                                         ker_num     = self.para.enc_bl03_ker_num,
                                         padding     = self.para.enc_bl03_padding,
                                         stride      = self.para.enc_bl03_stride,
                                         is_act      = self.para.enc_bl03_is_act,
                                         act_type    = self.para.enc_bl03_act_type,
                                         is_batch    = self.para.enc_bl03_is_batch,
                                         mode        = mode,
                                         scope       = scope
                                        )
        #Pooling 03
        with tf.device('/gpu:0'), tf.variable_scope("ENC_bl03_pool"):  
            o_bl03_pool = f_sig_pool(i_tensor  = o_bl03_conv2d,
                                     p_type    = self.para.enc_bl03_p_type,
                                     p_padding = self.para.enc_bl03_p_padding,
                                     p_stride  = self.para.enc_bl03_p_stride,
                                     p_ksize   = self.para.enc_bl03_p_ksize
                                    ) 
        with tf.device('/gpu:0'), tf.variable_scope("ENC_bl03_drop"):  
            o_bl03_drop = f_sig_drop(i_tensor = o_bl03_pool,
                                     rate     = self.para.enc_bl03_drop_prob,
                                     mode     = mode
                                    )
            o_bl03_output = o_bl03_drop   

        #========= Block 04
        #Conv04 
        with tf.device('/gpu:0'), tf.variable_scope("ENC_bl04_conv2d") as scope:  
            o_bl04_conv2d = f_sig_inct(i_tensor    = o_bl03_output,
                                         #ker_h       = self.para.enc_bl04_ker_h,
                                         #ker_w       = self.para.enc_bl04_ker_w,
                                         pre_ker_num = self.para.enc_bl04_pre_ker_num,
                                         ker_num     = self.para.enc_bl04_ker_num,
                                         padding     = self.para.enc_bl04_padding,
                                         stride      = self.para.enc_bl04_stride,
                                         is_act      = self.para.enc_bl04_is_act,
                                         act_type    = self.para.enc_bl04_act_type,
                                         is_batch    = self.para.enc_bl04_is_batch,
                                         mode        = mode,
                                         scope       = scope
                                        )
        #Global Pooling 04
        with tf.device('/gpu:0'), tf.variable_scope("ENC_bl04_pool"):  
            o_bl04_pool = f_sig_pool(i_tensor  = o_bl04_conv2d,
                                     p_type    = self.para.enc_bl04_p_type,
                                     p_padding = self.para.enc_bl04_p_padding,
                                     p_stride  = self.para.enc_bl04_p_stride,
                                     p_ksize   = self.para.enc_bl04_p_ksize
                                    ) 

        with tf.device('/gpu:0'), tf.variable_scope("ENC_bl04_drop"):  
            o_bl04_drop = f_sig_drop(i_tensor = o_bl04_pool,
                                     rate     = self.para.enc_bl04_drop_prob,
                                     mode     = mode
                                    )
            o_bl04_output = o_bl04_drop   

        ##======== Block 05/
        #with tf.device('/gpu:0'), tf.variable_scope("ENC_bl05_fc"):  
        #    o_bl05_fc = f_sig_fully_connected(i_tensor    = o_bl04_output,
        #                                      input_size  = self.para.enc_bl05_fc_input_size,
        #                                      output_size = self.para.enc_bl05_fc_output_size,
        #                                      is_act      = self.para.enc_bl05_fc_is_act,
        #                                      act_func    = self.para.enc_bl05_fc_act_func
        #                                     )
        #with tf.device('/gpu:0'), tf.variable_scope("ENC_bl05_drop"):  
        #    o_bl05_drop = f_sig_drop(i_tensor = o_bl05_fc,
        #                             rate     = self.para.enc_bl05_drop_prob,
        #                             mode     = mode
        #                            )

        ##======== Block 06/
        #with tf.device('/gpu:0'), tf.variable_scope("ENC_bl06_fc"):  
        #    o_bl06_fc = f_sig_fully_connected(i_tensor    = o_bl05_drop,
        #                                      input_size  = self.para.enc_bl06_fc_input_size,
        #                                      output_size = self.para.enc_bl06_fc_output_size,
        #                                      is_act      = self.para.enc_bl06_fc_is_act,
        #                                      act_func    = self.para.enc_bl06_fc_act_func
        #                                     )
        ##======== Block 07/
        #with tf.device('/gpu:0'), tf.variable_scope("ENC_bl07_fc"):  
        #    o_bl07_fc = f_sig_fully_connected(i_tensor    = o_bl06_fc,
        #                                      input_size  = self.para.enc_bl07_fc_input_size,
        #                                      output_size = self.para.enc_bl07_fc_output_size,
        #                                      is_act      = self.para.enc_bl07_fc_is_act,
        #                                      act_func    = self.para.enc_bl07_fc_act_func
        #                                     )

        return o_bl04_output #, o_bl06_fc  #4 (class num), 256 (latent dim)
