import numpy as np
import os
import re
import scipy.io

def get_pre_train_data(save_dir, data_dir):  

    nF = 128 
    nT = 256
    
    org_class_list = os.listdir(data_dir)
    class_list = []  #remove .file
    for nClass in range(0,len(org_class_list)):
        isHidden=re.match("\.",org_class_list[nClass])
        if (isHidden is None):
            class_list.append(org_class_list[nClass])
    class_num  = len(class_list)
    class_list = sorted(class_list)
    
    for nClass in range(class_num):
        # 01/ Expected class
        expectedClass = np.zeros([1,class_num])
        expectedClass[0,nClass] = 1
    
        # 02/ Handle file directory 
        class_name = class_list[nClass]  
        class_save_dir = save_dir + class_name
        if not os.path.exists(class_save_dir):
            os.makedirs(class_save_dir)
        print('\n===========  Start extracting all files in class {} Output [{}]\n'.format(class_name, class_save_dir))  
    
    
        input_dir = data_dir + class_name
        org_file_list = os.listdir(input_dir)
        file_list = []  #remove .file
        for nFile in range(0,len(org_file_list)):
            isHidden=re.match("\.",org_file_list[nFile])
            if (isHidden is None):
                file_list.append(org_file_list[nFile])
        file_num  = len(file_list)
        file_list = sorted(file_list)
        
        #--------------------------------------------------------------------------------  
        for nFile in range(file_num):
            nImage=0
            # 01. Reading file
            file_name = file_list[nFile]
            file_open = input_dir + '/' + file_name
            file_save = class_save_dir + '/' + file_name
            
            gam_str  = scipy.io.loadmat(file_open)
            data_str = gam_str['final_data']
            [nFreq, nTime] = np.shape(data_str)

            feat_num = np.floor(nTime/nT) 
            for m in range(int(feat_num)):
                tStart = m*nT  
                tStop  = tStart + nT
                one_image = data_str[:,tStart:tStop]
                [row_num, col_num] = np.shape(one_image)
                one_image = np.reshape(one_image,[1,row_num,col_num])
                if (m == 0):
                   seq_x = one_image
                   seq_y = expectedClass
                else:            
                   seq_x = np.concatenate((seq_x, one_image), axis=0)  
                   seq_y = np.concatenate((seq_y,expectedClass), axis=0)  

                # for test  
                if(np.size(seq_x[nImage,:,:]) != nT*nF):  
                    print('ERROR: Frame {} [{}:{}] of {} [exit]\n'.format(m,tStart,tStop, nTime))               
                    np.shape(seq_x[nImage,:,:])  
                    exit()
                nImage=nImage+1  
            np.savez(file_save, seq_x=seq_x, seq_y=seq_y)  
        print('Done extracting pre_train features for class {}: [{}] \n'.format(nClass, class_name)) 
    print('\n============================== Done extracting pre_train features \n')  
    
