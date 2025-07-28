from torch import tensor,float32
from torch.utils import data
import os
import numpy as np
import re
from scipy.io import loadmat
import pandas as pd
import mne
from autoreject import AutoReject

# eeg conf
xls_path = 'F:\\PLL\\静息态数据没有ICA\\意识障碍患者诊断结果.xls'
sfreq= 250
ch_names = ['Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 
            'O2', 'F7', 'F8', 'T7', 'T8', 'P7', 'P8', 'Fz', 'Cz', 
            'Pz', 'Lz', 'FC1', 'FC2', 'CP1', 'CP2', 'FC5', 'FC6', 
            'CP5', 'CP6', 'TP9', 'TP10', 'F1', 'F2', 'C1', 'C2', 
            'P1', 'P2', 'AF3', 'AF4', 'FC3', 'FC4', 'CP3', 'CP4', 
            'PO3', 'PO4', 'F5', 'F6', 'C5', 'C6', 'P5', 'P6', 'AF7', 
            'AF8', 'FT7', 'FT8', 'TP7', 'TP8', 'PO7', 'PO8', 'Fpz', 
            'CPz', 'POz', 'Oz']
locs_path = 'F:\\PLL\\62.xyz'
# ar conf
n_interpolates = np.array([1, 4])
consensus_percs = np.linspace(0, 1.0, 11)


        
def checkLabel(name,xls_path,type_label_dict):
    """输入片段文件名，输出此文件的类型名

    Args:
        name (str): 片段文件名

    Returns:
        str: 文件type
    """
    # hc
    matchObj = re.match( r'([0-9]*)data.mat', name, re.M|re.I)     # 正则表达式匹配文件名
    if matchObj:
        type = 'HC'
        subject_name = matchObj.group(1)
        head = "完整"
    else:
        # pdoc
        df = pd.read_excel(xls_path)
        pdoc_result = df.set_index('姓名' )['意识诊断'].to_dict()
        head_result = df.set_index('姓名' )['颅骨完整程度'].to_dict()
        matchObj = re.match( r'([0-9]*)-(.*)-(.*).mat', name, re.M|re.I)     # 正则表达式匹配文件名，提取患者姓名
        if matchObj and matchObj.group(2) in pdoc_result:
            subject_name = matchObj.group(2)
            type =  pdoc_result[subject_name]
            head = head_result[subject_name]
        else:
            return -1,-1,-1
    if not type in type_label_dict:
        return -1,-1,-1
    return type_label_dict[type],subject_name,head

def getData(data_filepath):
    # 读取mat文件
    raw_mat = loadmat(data_filepath)
    raw_data = raw_mat['data']
    return raw_data


class MDataset(data.Dataset):
    def __init__(self,file_path,window_duration,window_overlap,chosen_chs,type_label_dict):
        # parse name
        label,subject_name,head  = checkLabel(os.path.basename(file_path),xls_path,type_label_dict)
        if label == -1:
            self.fail_init = True
            return
        else:
            self.fail_init = False
        # 赋值
        self.subject_name = subject_name
        self.head = head
        self.Y = label
        # epochs
        fif_path = f'temp//{subject_name}-{window_duration}-{window_overlap}-epo.fif'
        if os.path.isfile(fif_path):
            # 文件存在，读取Epochs
            epochs = mne.read_epochs(fif_path,verbose=False)
        else:
            rawdata = getData(file_path)
            # eeg preprocess
            info = mne.create_info(ch_names, sfreq, ch_types='eeg', verbose=None)
            montage = mne.channels.read_custom_montage(locs_path)
            raw = mne.io.RawArray(rawdata, info,first_samp=0, copy='auto', verbose=None)
            epochs = mne.make_fixed_length_epochs(raw, duration= window_duration, preload=True, overlap=window_overlap, id=1, verbose=None)
            epochs.set_montage(montage,match_alias=True)
            picks = mne.pick_types(raw.info, eeg=True, stim=False, eog=False,include=ch_names, exclude=[])
            ar = AutoReject(n_interpolates, consensus_percs,picks = picks,cv=5,n_jobs=4,thresh_method='random_search', random_state=42,verbose =False)
            epochs,ar_log = ar.fit_transform(epochs,return_log=True)
            epochs.save(fif_path, overwrite=True) 
            ar_log.save(f'temp//{subject_name}-{window_duration}-{window_overlap}-rejection.npz',overwrite=True)
        self.epochs = epochs.pick(chosen_chs)

    def __getitem__(self,index):
        x = self.epochs.get_data()[index]
        x = tensor(x,dtype=float32)
        y = tensor(self.Y)
        return x,y

    def __len__(self):
        if self.fail_init:
            return 0
        return len(self.epochs)

