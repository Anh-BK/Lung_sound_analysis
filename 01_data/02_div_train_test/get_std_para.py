import sys
import os
import re
import numpy as np
import scipy.io

#----- For cycle 
data_dir   =  "./../12_split_gam_10s_icb/data_train/"
para_file  = "stat_cyc_para"

#-------------------------------------------------------------
# Get list of class
org_class_name_list = os.listdir(data_dir)
class_name_list = []
for i in range(0,len(org_class_name_list)):
   isHidden=re.match("\.",org_class_name_list[i])
   if (isHidden is None):
      class_name_list.append(org_class_name_list[i])
class_name_list = sorted(class_name_list)        
class_name_num  = len(class_name_list)

#print class_name_list
#exit()

# For every class
total_col = 0
for nClass in range(0, class_name_num): 
#for nClass in range(0, 1): 
    #3.1 Collect the file Name List
    class_name  = class_name_list[nClass]
    class_data_open  = data_dir + class_name + '/'

    org_file_name_list = os.listdir(class_data_open)
    file_name_list = []  #remove .file
    for i in range(0,len(org_file_name_list)):
       isHidden=re.match("\.",org_file_name_list[i])
       if (isHidden is None):
          file_name_list.append(org_file_name_list[i])
    file_name_list = sorted(file_name_list)
    file_name_num  = len(file_name_list)

    #print file_name_list
    #exit()

    # For every file in class
    for nFile in range(0, file_name_num):
    #for nFile in range(0, 3):
       file_name = os.path.splitext(file_name_list[nFile])[0]

       file_data_open = class_data_open + file_name + '.mat'
       #print (file_data_open)
       #exit()

       gam_str    = scipy.io.loadmat(file_data_open)
       data_matrix = gam_str['final_data']
       #print(np.shape(data_matrix))
       #exit()

       [row, col] = np.shape(data_matrix)
       total_col = total_col + col


       if(nFile == 0):
           data_sum_f  = np.sum(data_matrix,1)
           data_sum_f2 = np.sum(np.square(data_matrix),1)
       else:
           data_sum_f  = data_sum_f + np.sum(data_matrix,1)
           data_sum_f2 = data_sum_f2 + np.sum(np.square(data_matrix),1)
       
       #print (np.shape(data_sum_f))
       #print (np.shape(data_sum_f2))
       #exit()

data_mean = data_sum_f/total_col

data_var = abs(data_sum_f2/total_col - np.square(data_mean)) 

data_var = np.sqrt(data_var) 

np.savez(para_file, data_mean=data_mean, data_var=data_var)
print(np.shape(data_mean))          
print(np.shape(data_var))          


