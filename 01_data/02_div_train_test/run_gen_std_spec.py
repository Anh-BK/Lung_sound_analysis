import os
from gen_std_spec import *
#from get_test_para import *

data_dir   =  "./../11_gam_pa_2/"

store_dir       = "./../11_gam_pa_2_nor/"

if not os.path.exists(store_dir):
    os.makedirs(store_dir)

gen_std_spec(data_dir, store_dir)
