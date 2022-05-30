# This code is modified from https://github.com/facebookresearch/low-shot-shrink-hallucinate

import torch
from PIL import Image
import numpy as np
import torchvision.transforms as transforms
import data.additional_transforms as add_transforms
from data.dataset import SimpleDataset, SetDataset, EpisodicBatchSampler
from abc import abstractmethod

class TransformLoader:
    def __init__(self, image_size, 
                 normalize_param    = dict(mean= [0.485, 0.456, 0.406] , std=[0.229, 0.224, 0.225]),
                 jitter_param       = dict(Brightness=0.4, Contrast=0.4, Color=0.4)):
        self.image_size = image_size
        self.resize_size = 92
        self.normalize_param = normalize_param
        self.jitter_param = jitter_param
    
    def parse_transform(self, transform_type):
        if transform_type=='ImageJitter':
            method = add_transforms.ImageJitter( self.jitter_param )
            return method
        method = getattr(transforms, transform_type)
        if transform_type=='RandomResizedCrop':
            return method(self.image_size) 
        elif transform_type=='CenterCrop':
            return method(self.image_size) 
        elif transform_type=='Resize':
            return method((int(self.resize_size), int(self.resize_size)))
        elif transform_type=='Normalize':
            return method(**self.normalize_param )
        else:
            return method()

    def get_composed_transform(self, aug = False):
        if aug:
            transform_list = ['RandomResizedCrop', 'RandomHorizontalFlip', 'ToTensor', 'Normalize']
        else:
            transform_list = ['Resize','CenterCrop', 'ToTensor', 'Normalize']

        transform_funcs = [ self.parse_transform(x) for x in transform_list]
        transform = transforms.Compose(transform_funcs)
        return transform

class DataManager:
    @abstractmethod
    def get_data_loader(self, data_file, aug):
        pass 


class SimpleDataManager(DataManager):
    def __init__(self, image_size, batch_size):        
        super(SimpleDataManager, self).__init__()
        self.batch_size   = batch_size
        self.trans_loader = TransformLoader(image_size)

    def get_data_loader(self, data_file, aug): #parameters that would change on train/val set
        transform = self.trans_loader.get_composed_transform(aug)
        dataset = SimpleDataset(data_file, transform) # number of workers
        data_loader_params = dict(batch_size = self.batch_size, shuffle = True, num_workers = 12, pin_memory =  False)       
        data_loader = torch.utils.data.DataLoader(dataset, **data_loader_params)

        return data_loader

class SetDataManager(DataManager):
    def __init__(self, d_name, image_size, n_way, n_support, n_query, n_eposide = 200):        
        super(SetDataManager, self).__init__()
        self.image_size = image_size
        self.n_way      = n_way
        self.batch_size = n_support + n_query
        self.n_eposide  = n_eposide

        ##
        # Augmentation used in recent work RENet 
        # https://github.com/dahyun-kang/renet
        #
        if d_name == 'cifarfs':
            normalize_param = dict(mean= [0.5071, 0.4866, 0.4409], std = [0.2009, 0.1984, 0.2023])
        else:
            normalize_param = dict(mean= np.array([x / 255.0 for x in [125.3, 123.0, 113.9]]), std = np.array([x / 255.0 for x in [63.0, 62.1, 66.7]]))

        self.trans_loader = TransformLoader(image_size=image_size, normalize_param=normalize_param)
        

    def get_data_loader(self, data_file, aug): #parameters that would change on train/val set
        transform = self.trans_loader.get_composed_transform(aug)
        print('TRANSFORMS : ', transform)
        dataset = SetDataset( data_file , self.batch_size, transform )
        sampler = EpisodicBatchSampler(len(dataset), self.n_way, self.n_eposide ) 
        data_loader_params = dict(batch_sampler = sampler,  num_workers = 12, pin_memory = False)       
        data_loader = torch.utils.data.DataLoader(dataset, **data_loader_params)
        return data_loader
