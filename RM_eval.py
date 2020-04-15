import numpy as np
import random
import os
from tqdm import tqdm
import copy


def create_folder(folder_name):
    if not os.path.exists(folder_name):
        print('{} folder does not exist, creating it.'.format(folder_name))
        os.makedirs(folder_name)


dataset = 'foa'     # ['foa', 'mic']
trial_num = 1

normal_feature_path = '/home/zhangkeming/dcase2019/task3-dataset/feat_label_mel/{}_dev_norm'.format(dataset)
label_path = '/home/zhangkeming/dcase2019/task3-dataset/feat_label_mel/{}_dev_label'.format(dataset)
write_feature_path = '/home/zhangkeming/dcase2019/task3-dataset/feat_label_mel/{}_dev_norm_trial{}'.format(dataset, trial_num)

channel=4
augment_num = 2            # number of augmented files for each original file
freq_num = 128             # number of frequency bins, 1024 for STFT and 128 for Mel
F_ratio = 0.3              #0.2
F = int(freq_num * F_ratio) # maximum number of frequency bins that could be dropped
frame_num = 3000            # number of frames within a file
T_ratio = 0.04
T = int(frame_num * T_ratio)# maximum number of frames that could be dropped
filenames_list = list()
test_splits = [1, 2, 3, 4]
policy_list = [['time'], ['freq'], ['time', 'freq']]

new_split_list = [[21, 22, 23, 24], [25, 26, 27, 28], [29, 30, 31, 32]] 
for filename in os.listdir(normal_feature_path):
    if filename[6].isdigit():
        continue
    else:
        split_num = int(filename[5])
        if split_num in test_splits: # check which split the file belongs to
            filenames_list.append(filename)

pbar = tqdm(total=len(filenames_list))
for file_cnt, file_name in enumerate(filenames_list):
    feat_file = np.load(os.path.join(normal_feature_path, file_name))
    label_file = np.load(os.path.join(label_path, file_name))
    label_sed=label_file[:,0:11]
    data_list = list()
    first_frame=0
    sin_cur=False
    for t_label in range(3000):
        if(label_sed[t_label].max()>=1):#first frame

            if t_label == 0:
                sin_cur = True
            elif not sin_cur:
                if (t_label > 1):
                    sin_cur = True
                    data_list.append([first_frame, t_label - 1])
        elif (label_sed[t_label].max() == 0):  # first frame

            if t_label == 0:
                sin_cur = False
            elif  sin_cur:
                sin_cur = False
                if (t_label > 1):
                    first_frame = t_label
                    sin_cur = False
    
    file_split = file_name.split('_')

    split_name = file_split[0]
    split_num = int(split_name[5])
    file_num = int(file_split[-1][:-4])
    file_inter= file_name[6:15]
    if ((split_num-1)*25+1 <= file_num and file_num <= split_num*25):
        continue


    T_len=30
    for j in range(augment_num):
        temp = copy.deepcopy(feat_file)
        for i in range(3):  
            policy = policy_list[i]
            new_split = new_split_list[i]  
            temp = copy.deepcopy(feat_file)             
            if 'time' in policy:
                # 1 randomly drop frames
                for tnum in range(channel):
                    for t_repeat in range(2):
                        data_noise=random.choice(data_list)
                        data_start=data_noise[0]
                        data_end=data_noise[1]
                        data_len = data_end - data_start
                        if data_len <= T_len :
                            temp[data_start:data_end, tnum * freq_num:(tnum + 1) * freq_num] *= np.random.randn(data_len, 128)
                        else:
                            t1=random.randint(0, data_len)
                            if (data_start+t1+T_len) <= data_end:
                                temp[(data_start+t1):(data_start+t1+T_len), tnum * freq_num:(tnum + 1) * freq_num] *=np.random.randn(T_len, 128)
                            elif (data_start+t1+T_len) > data_end:
                                temp[(data_start+t1):data_end,tnum * freq_num:(tnum + 1) * freq_num] *= np.random.randn(data_len-t1, 128)                                
                                temp[data_start:(2*data_start+T_len+t1-data_end),tnum * freq_num:(tnum + 1) * freq_num] *= np.random.randn((t1+T_len-data_len), 128)
                                
            if 'freq' in policy:
                # randomly drop frequency bins
                # first channel
                for fnum in range(channel):
                    for t_repeat in range(1):
                        f = random.randint(0, F)      # length of the dropped band
                        f0 = random.randint(fnum*freq_num, (fnum+1)*freq_num - f) # start point of the band
                        t = random.randint(0, len(data_list))    # length of the dropped chunk
                        mask_data=np.ones([3000, f],int)
                        for index_data in range(len(data_list)):
                            data_len=data_list[index_data][1]-data_list[index_data][0]
                            mask_data[data_list[index_data][0]:data_list[index_data][1]] =np.random.randn(data_len, f)
                        temp[:, f0:f0 + f] *= mask_data
            new_file_num = j* 100 + file_num
            np.save(os.path.join(write_feature_path, 'split' + str(new_split[split_num-1]) + file_inter+str(new_file_num)+'.npy'), temp)
            
    pbar.update(1)
pbar.close()