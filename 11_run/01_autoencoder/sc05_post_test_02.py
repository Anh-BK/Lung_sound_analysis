import numpy as np
import os
import argparse
import math
import scipy.io
import re
import time
import datetime
from scipy import stats


#======================================================  01/ DIRECTORY 
# TODO-Dir
out_dir    = "./data/"
source_dir = "./data/02_mid_test_extr/" 

source_dir_02 = "./../../12_run_inception/05_cnn_inc/data/02_mid_test_extr/" 

#For storing data for train 02
stored_dir = os.path.abspath(os.path.join(os.path.curdir,out_dir))
if not os.path.exists(stored_dir):
    os.makedirs(stored_dir)
des_dir = os.path.abspath(os.path.join(stored_dir, "04_post_test_data_concat")) 
if not os.path.exists(des_dir):
    os.makedirs(des_dir)


#======================================================  03/ HANDLE SOURCE FILE
source_dir = os.path.abspath(source_dir) 
org_source_file_list = os.listdir(source_dir)
source_file_list = []  #remove .file
for nFileSc in range(0,len(org_source_file_list)):
    isHidden=re.match("\.",org_source_file_list[nFileSc])
    if (isHidden is None):
        source_file_list.append(org_source_file_list[nFileSc])
source_file_num  = len(source_file_list)
source_file_list = sorted(source_file_list)


#======================================================  04/ COLLECT DATA into 22 GROUP
for nFile in range(int(source_file_num)):

    source_file_name = source_file_list[nFile]
    des_file = os.path.abspath(os.path.join(des_dir, source_file_name))

    # Open file
    file_open = source_dir + '/' + source_file_name
    file_str  = np.load(file_open)   #31x256
  
    file_open_02 = source_dir_02 + '/' + source_file_name
    file_str_02  = np.load(file_open_02)   #31x256

    seq_x  =  np.concatenate((file_str, file_str_02), axis = 1)

    # Save file
    np.save(des_file, seq_x)
