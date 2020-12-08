import sys
import os
import re
import numpy as np
import shutil

data_dir  =  "./../11_mel_pa/"
save_dir  =  "./../12_split_mel_pa/"

#data_dir  =  "./../11_2spec/"
#save_dir  =  "./../12_split_2spec_pa/"

save_train_dir = save_dir + 'data_train'
save_test_dir  = save_dir + 'data_test'

if not os.path.exists(save_dir):
    os.makedirs(save_dir)

if not os.path.exists(save_train_dir):
    os.makedirs(save_train_dir)

if not os.path.exists(save_test_dir):
    os.makedirs(save_test_dir)


org_folder_name_list = os.listdir(data_dir)
folder_name_list = []  #remove .folder
for i in range(0,len(org_folder_name_list)):
   isHidden=re.match("\.",org_folder_name_list[i])
   if (isHidden is None):
      folder_name_list.append(org_folder_name_list[i])
folder_name_list = sorted(folder_name_list)
folder_name_num  = len(folder_name_list)

# For every file in class
for nFolder in range(0, folder_name_num):
    folder_name  = folder_name_list[nFolder]
    folder_dir   = data_dir + folder_name

    save_folder_train_dir = save_train_dir + '/' + folder_name
    if not os.path.exists(save_folder_train_dir):
        os.makedirs(save_folder_train_dir)


    org_file_name_list = os.listdir(folder_dir)
    file_name_list = []  #remove .file
    for i in range(0,len(org_file_name_list)):
       isHidden=re.match("\.",org_file_name_list[i])
       if (isHidden is None):
          file_name_list.append(org_file_name_list[i])
    file_name_list = sorted(file_name_list)
    file_name_num  = len(file_name_list)
    
    test_num  = int(file_name_num*0.2)
    train_num = int(file_name_num - test_num)
    
    #kk = np.random.permutation(file_name_num)
    file_name_shuffle = 'pa_kk_' + str(nFolder) + '.npy'
    #np.save(file_name_shuffle, kk)
    kk = np.load(file_name_shuffle)

    for nFile in range(file_name_num):
        #if(nFile >= file_name_num - 1*test_num): #fold-5
        #if((nFile >= file_name_num - 2*test_num) and (nFile < file_name_num - 1*test_num)): #fold-4
        if((nFile >= file_name_num - 3*test_num) and (nFile < file_name_num - 2*test_num)): #fold-3
        #if((nFile >= file_name_num - 4*test_num) and (nFile < file_name_num - 3*test_num)): #fold-2
        #if(nFile < file_name_num - 4*test_num): #fold-1
            file_save_dir = save_test_dir +  '/' + folder_name + '_' + file_name_list[kk[nFile]]
        else:
            file_save_dir = save_folder_train_dir + '/' + folder_name + '_' + file_name_list[kk[nFile]]

        file_cp = folder_dir + '/' + file_name_list[kk[nFile]]
        
        shutil.copy(file_cp, file_save_dir)
        #print(file_cp)
        #print(file_save_dir)
