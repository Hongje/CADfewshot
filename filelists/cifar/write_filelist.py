import numpy as np
from os import listdir
from os.path import isfile, isdir, join
import os

data_path = '/path/to/dataset/.............../cifar_fs'
savedir   = './'
dataset_list = ['meta-train','meta-val','meta-test']

folder_list = dict()
label_dict  = dict()
for d_split in dataset_list:
    folder_list[d_split] = [f for f in listdir(join(data_path,d_split)) if isdir(join(data_path, d_split, f))]
    folder_list[d_split].sort()
    label_dict[d_split] = dict(zip(folder_list[d_split],range(0,len(folder_list[d_split]))))
    print(f'{d_split}  {len(folder_list[d_split])}')

classfile_list_all = dict()
for d_split in dataset_list:
    classfile_list_all[d_split] = []

for d_split in dataset_list:
    for i, folder in enumerate(folder_list[d_split]):
        folder_path = join(data_path, d_split, folder)
        classfile_list_all[d_split].append( [ join(folder_path, cf) for cf in listdir(folder_path) if (isfile(join(folder_path,cf)) and cf[0] != '.')])

save_name = {
    'meta-train': 'base',
    'meta-val'  : 'val',
    'meta-test' : 'novel'
}

for dataset in dataset_list:
    file_list  = []
    label_list = []
    for i, classfile_list in enumerate(classfile_list_all[dataset]):
        file_list  = file_list + classfile_list
        label_list = label_list + np.repeat(i, len(classfile_list)).tolist()
    
    
    fo = open(savedir + save_name[dataset] + ".json", "w")
    fo.write('{"label_names": [')
    fo.writelines(['"%s",' % item  for item in folder_list[dataset]])
    fo.seek(0, os.SEEK_END) 
    fo.seek(fo.tell()-1, os.SEEK_SET)
    fo.write('],')

    fo.write('"image_names": [')
    fo.writelines(['"%s",' % item  for item in file_list])
    fo.seek(0, os.SEEK_END) 
    fo.seek(fo.tell()-1, os.SEEK_SET)
    fo.write('],')

    fo.write('"image_labels": [')
    fo.writelines(['%d,' % item  for item in label_list])
    fo.seek(0, os.SEEK_END) 
    fo.seek(fo.tell()-1, os.SEEK_SET)
    fo.write(']}')

    fo.close()
    print("%s -OK" %dataset)
