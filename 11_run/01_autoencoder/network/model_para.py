class model_para(object):

    def __init__(self):

        #======================================================== INPUT PARAMETER
        self.n_class                 = 4
        self.l2_lamda                = 0.0001

        self.i_nF                    = 128
        self.i_nT                    = 256
        self.i_nC                    = 1

        #========================================================= DECODER

        #========= BL01
        #Fully connected
        self.dec_bl01_fc_input_size  = 256
        self.dec_bl01_fc_output_size = 8*16*256
        self.dec_bl01_fc_is_act      = True
        self.dec_bl01_fc_act_func    = 'RELU'

        #Reshape
        self.dec_bl01_nF             = 8
        self.dec_bl01_nT             = 16
        self.dec_bl01_nC             = 256

        #========= BL02
        #conv
        self.dec_bl02_ker_h          = 3
        self.dec_bl02_ker_w          = 3
        self.dec_bl02_pre_ker_num    = 256
        self.dec_bl02_ker_num        = 128
        self.dec_bl02_padding        = 'SAME' 
        self.dec_bl02_stride         = [1,2,2,1]      
        self.dec_bl02_o_exp_h        = 16
        self.dec_bl02_o_exp_w        = 32
        #act
        self.dec_bl02_is_act         = True
        self.dec_bl02_act_type       = 'RELU'
        #batch
        self.dec_bl02_is_batch       = False

        #========= BL03
        #conv
        self.dec_bl03_ker_h          = 3
        self.dec_bl03_ker_w          = 3
        self.dec_bl03_pre_ker_num    = 128
        self.dec_bl03_ker_num        = 64
        self.dec_bl03_padding        = 'SAME' 
        self.dec_bl03_stride         = [1,2,2,1]      
        self.dec_bl03_o_exp_h        = 32
        self.dec_bl03_o_exp_w        = 64
        #act
        self.dec_bl03_is_act         = True
        self.dec_bl03_act_type       = 'RELU'
        #batch
        self.dec_bl03_is_batch       = False

        #========= BL04
        #conv
        self.dec_bl04_ker_h          = 3
        self.dec_bl04_ker_w          = 3
        self.dec_bl04_pre_ker_num    = 64
        self.dec_bl04_ker_num        = 32
        self.dec_bl04_padding        = 'SAME' 
        self.dec_bl04_stride         = [1,2,2,1]      
        self.dec_bl04_o_exp_h        = 64
        self.dec_bl04_o_exp_w        = 128
        #act
        self.dec_bl04_is_act         = True
        self.dec_bl04_act_type       = 'RELU'
        #batch
        self.dec_bl04_is_batch       = False

        #========= BL05
        #conv
        self.dec_bl05_ker_h          = 3
        self.dec_bl05_ker_w          = 3
        self.dec_bl05_pre_ker_num    = 32
        self.dec_bl05_ker_num        = 1
        self.dec_bl05_padding        = 'SAME' 
        self.dec_bl05_stride         = [1,2,2,1]      
        self.dec_bl05_o_exp_h        = 128
        self.dec_bl05_o_exp_w        = 256
        #act
        self.dec_bl05_is_act         = True
        self.dec_bl05_act_type       = 'TANH'
        #batch
        self.dec_bl05_is_batch       = False

        #========================================================== ENCODER

        #========= BL01
        #conv
        self.enc_bl01_ker_h          = 3
        self.enc_bl01_ker_w          = 3
        self.enc_bl01_pre_ker_num    = 1
        self.enc_bl01_ker_num        = 32
        self.enc_bl01_padding        = 'SAME' 
        self.enc_bl01_stride         = [1,1,1,1]      
        #act
        self.enc_bl01_is_act         = True
        self.enc_bl01_act_type       = 'ELU'
        #batch
        self.enc_bl01_is_batch       = True
        #pool
        self.enc_bl01_is_pool        = True  
        self.enc_bl01_p_type         = 'MAX' 
        self.enc_bl01_p_padding      = 'VALID' 
        self.enc_bl01_p_stride       = [1,2,2,1]
        self.enc_bl01_p_ksize        = [1,2,2,1]
        ##drop
        self.enc_bl01_is_drop        = True
        self.enc_bl01_drop_prob      = 0.1

        #========= BL02
        #conv
        self.enc_bl02_ker_h          = 3
        self.enc_bl02_ker_w          = 3
        self.enc_bl02_pre_ker_num    = 32
        self.enc_bl02_ker_num        = 64
        self.enc_bl02_padding        = 'SAME' 
        self.enc_bl02_stride         = [1,1,1,1]      
        #act
        self.enc_bl02_is_act         = True
        self.enc_bl02_act_type       = 'ELU'
        #batch
        self.enc_bl02_is_batch       = True
        #pool
        self.enc_bl02_is_pool        = True  
        self.enc_bl02_p_type         = 'MAX' 
        self.enc_bl02_p_padding      = 'VALID' 
        self.enc_bl02_p_stride       = [1,2,2,1]
        self.enc_bl02_p_ksize        = [1,2,2,1]
        ##drop
        self.enc_bl02_is_drop        = True
        self.enc_bl02_drop_prob      = 0.15

        #========= BL03
        #conv
        self.enc_bl03_ker_h          = 3
        self.enc_bl03_ker_w          = 3
        self.enc_bl03_pre_ker_num    = 64
        self.enc_bl03_ker_num        = 128
        self.enc_bl03_padding        = 'SAME' 
        self.enc_bl03_stride         = [1,1,1,1]      
        #act
        self.enc_bl03_is_act         = True
        self.enc_bl03_act_type       = 'ELU'
        #batch
        self.enc_bl03_is_batch       = True
        #pool
        self.enc_bl03_is_pool        = True  
        self.enc_bl03_p_type         = 'MAX' 
        self.enc_bl03_p_padding      = 'VALID' 
        self.enc_bl03_p_stride       = [1,2,2,1]
        self.enc_bl03_p_ksize        = [1,2,2,1]
        ##drop
        self.enc_bl03_is_drop        = True
        self.enc_bl03_drop_prob      = 0.2

        #========= BL04
        #conv
        self.enc_bl04_ker_h          = 3
        self.enc_bl04_ker_w          = 3
        self.enc_bl04_pre_ker_num    = 128
        self.enc_bl04_ker_num        = 256
        self.enc_bl04_padding        = 'SAME' 
        self.enc_bl04_stride         = [1,1,1,1]      
        #act
        self.enc_bl04_is_act         = True
        self.enc_bl04_act_type       = 'ELU'
        #batch
        self.enc_bl04_is_batch       = True
        #pool
        self.enc_bl04_is_pool        = True  
        self.enc_bl04_p_type         = 'GLOBAL_MAX' 
        self.enc_bl04_p_padding      = 'VALID' 
        self.enc_bl04_p_stride       = [1,2,2,1]
        self.enc_bl04_p_ksize        = [1,2,2,1]
        ##drop
        self.enc_bl04_is_drop        = True
        self.enc_bl04_drop_prob      = 0.25

        #========= BL05
        #Fully connected
        self.enc_bl05_fc_input_size  = 256
        self.enc_bl05_fc_output_size = 1024
        self.enc_bl05_fc_is_act      = True
        self.enc_bl05_fc_act_func    = 'RELU'
        self.enc_bl05_is_drop        = True
        self.enc_bl05_drop_prob      = 0.3

        #========= BL07
        #Fully connected
        self.enc_bl06_fc_input_size  = 1024
        self.enc_bl06_fc_output_size = self.n_class
        self.enc_bl06_fc_is_act      = False
        self.enc_bl06_fc_act_func    = 'RELU'

