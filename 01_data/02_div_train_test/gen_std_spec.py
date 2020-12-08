import sys
import os
import re
import numpy as np
import scipy.io

#from matplotlib.pyplot import imshow
#import matplotlib.pyplot as plt

def gen_std_spec(data_dir, store_dir):
 


   file_para = np.load('./para.npz')

   gam_mean = file_para['data_mean']
   gam_var = file_para['data_var']

   gam_mean = np.reshape(gam_mean, (64,1))
   gam_var  = np.reshape(gam_var , (64,1)) 

   #print(np.shape(gam_mean))
   #print(np.shape(gam_var))
   #exit()

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
   for nClass in range(0, class_name_num): 
   #for nClass in range(0, 1): 
       #3.1 Collect the file Name List
       class_name  = class_name_list[nClass]
       class_data_open  = data_dir + class_name + '/'

       class_store = store_dir + '/' + class_name 
       if not os.path.exists(class_store):
          os.makedirs(class_store)

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
          gam_matrix = gam_str['final_data']
          gam_matrix = (gam_matrix - gam_mean)/gam_var
          #print(np.shape(gam_matrix))
          #exit()

          #Store std file
          des_file = class_store + '/' + file_name
          np.save(des_file, gam_matrix)


