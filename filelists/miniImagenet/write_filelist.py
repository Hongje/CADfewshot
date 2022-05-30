import numpy as np
from os import listdir
from os.path import isfile, isdir, join
import os
import json
import random



data_path = '/............................../miniimagenet'
savedir = './'
dataset_list = ['train','val','test']
#

IMAGE_PATH = os.path.join(data_path, 'images')
SPLIT_PATH = os.path.join(data_path, 'split')

folder_list = dict()
label_dict  = dict()

for d_split in dataset_list:
    csv_path   = join(SPLIT_PATH,  d_split + '.csv')
    lines      = [x.strip() for x in open(csv_path, 'r').readlines()][1:]
    
    data  = []
    for l in lines:
        _, wnid = l.split(',')
        if wnid not in data:
            data.append(wnid)
            
    folder_list[d_split] = data
    label_dict[d_split]  = dict(zip(folder_list[d_split],range(0,len(folder_list[d_split]))))
    print(f'{d_split}  {len(folder_list[d_split])}')

file_list  = dict()
label_list = dict()

for d_split in dataset_list:
    csv_path   = join(SPLIT_PATH,  d_split + '.csv')
    lines      = [x.strip() for x in open(csv_path, 'r').readlines()][1:]
    
    data = []
    label = []
    lb = -1
    wnids = []

    for l in lines:
        name, wnid = l.split(',')
        path = join(IMAGE_PATH, name)
        if wnid not in wnids:
            wnids.append(wnid)
            lb += 1
        data.append(path)
        label.append(lb)
    
    
    file_list[d_split]  = data
    label_list[d_split] = label 
    print(f'{d_split}  {len(file_list[d_split])} {len(set(label_list[d_split]))}')
    

save_name = {
    'train': 'base',
    'val'  : 'val',
    'test' : 'novel'
}

for dataset in dataset_list:
    f_list = file_list[dataset]
    l_list = label_list[dataset]
    
    fo = open(savedir + save_name[dataset] + ".json", "w")
    fo.write('{"label_names": [')
    fo.writelines(['"%s",' % item  for item in folder_list[dataset]])
    fo.seek(0, os.SEEK_END) 
    fo.seek(fo.tell()-1, os.SEEK_SET)
    fo.write('],')

    fo.write('"image_names": [')
    fo.writelines(['"%s",' % item  for item in f_list])
    fo.seek(0, os.SEEK_END) 
    fo.seek(fo.tell()-1, os.SEEK_SET)
    fo.write('],')

    fo.write('"image_labels": [')
    fo.writelines(['%d,' % item  for item in l_list])
    fo.seek(0, os.SEEK_END) 
    fo.seek(fo.tell()-1, os.SEEK_SET)
    fo.write(']}')

    fo.close()
    print("%s -OK" %dataset)
