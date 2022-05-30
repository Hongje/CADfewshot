import warnings
warnings.filterwarnings("ignore")
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim
import torch.optim.lr_scheduler as lr_scheduler
import time
import os, random
import glob, sys

import configs
import backbone
from data.datamgr import SimpleDataManager, SetDataManager
from methods.baselinetrain import BaselineTrain
from methods.baselinefinetune import BaselineFinetune
# METHODS
from methods.cad import CAD
from methods.protonet import ProtoNet

#
from io_utils import model_dict, parse_args, get_resume_file  
from utils import print_network
from tqdm import tqdm

def train(base_loader, val_loader, model, optimization, start_epoch, stop_epoch, params):    
    if optimization == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=params.lr, weight_decay=params.decay)
    elif optimization == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=params.lr, momentum=0.9, nesterov=True, weight_decay=0.0005)
    else:
       raise ValueError('Unknown optimization, please define by yourself')

    
    # schduler
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=params.milestones, gamma=0.05)
    print(f'Training with {optimizer} \n')
    
    max_acc = 0
    pbar = tqdm(range(start_epoch, stop_epoch), ncols=80, desc=' ')
    for epoch in pbar:
        model.train()
        model.train_loop(epoch, base_loader,  optimizer , pbar) #model are called by reference, no need to return
        model.eval()

        if not os.path.isdir(params.checkpoint_dir):
            os.makedirs(params.checkpoint_dir)

        acc = model.test_loop( val_loader)
        if acc > max_acc : #for baseline and baseline++, we don't use validation in default and we let acc = -1, but we allow options to validate with DB index
            print("best model! save...")
            max_acc = acc
            outfile = os.path.join(params.checkpoint_dir, 'best_model.tar')
            torch.save({'epoch':epoch, 'state':model.state_dict()}, outfile)

        if (epoch % params.save_freq==0) or (epoch==stop_epoch-1):
            outfile = os.path.join(params.checkpoint_dir, '{:d}.tar'.format(epoch))
            torch.save({'epoch':epoch, 'state':model.state_dict()}, outfile)
        # 
        scheduler.step()
    return model

if __name__=='__main__':
    seed = 10
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    params = parse_args('train')


    if params.dataset == 'cross':
        base_file = configs.data_dir['miniImagenet'] + 'all.json' 
        val_file  = configs.data_dir['CUB'] + 'val.json' 
    else:
        base_file = configs.data_dir[params.dataset] + 'base.json' 
        val_file  = configs.data_dir[params.dataset] + 'val.json' 
         
    
    image_size = 84 


    if params.stop_epoch == -1: 
        if params.method in ['baseline', 'baseline++'] :
            if params.dataset in ['CUB']:
                params.stop_epoch = 200 # This is different as stated in the open-review paper. However, using 400 epoch in baseline actually lead to over-fitting
            elif params.dataset in ['miniImagenet', 'cross']:
                params.stop_epoch = 400
            else:
                params.stop_epoch = 400 #default
        else: #meta-learning methods
            if params.n_shot == 1:
                params.stop_epoch = 80
            elif params.n_shot == 5:
                params.stop_epoch = 60
                params.milestones = [40,50]
            else:
                params.stop_epoch = 600 #default
     

    if params.method in ['baseline', 'baseline++'] :
        base_datamgr    = SimpleDataManager(image_size, batch_size = 256) # 16
        base_loader     = base_datamgr.get_data_loader( base_file , aug = params.train_aug )
        val_datamgr     = SimpleDataManager(image_size, batch_size = 128)
        val_loader      = val_datamgr.get_data_loader( val_file, aug = False)

        if params.method == 'baseline':
            model           = BaselineTrain( model_dict[params.model], params.num_classes)
        elif params.method == 'baseline++':
            model           = BaselineTrain( model_dict[params.model], params.num_classes, loss_type = 'dist')


    elif params.method in ['cad', 'proto']:
        
        n_query = max(1, int(15* params.test_n_way/params.train_n_way)) #if test_n_way is smaller than train_n_way, reduce n_query to keep batch size small
        print(f'Number of queries {n_query} | Stop Epoch {params.stop_epoch} | Augmentation {params.train_aug} ')

        train_few_shot_params   = dict(n_way = params.train_n_way, n_support = params.n_shot)
        base_datamgr            = SetDataManager(params.dataset,image_size, n_query = n_query,  **train_few_shot_params)
        base_loader             = base_datamgr.get_data_loader( base_file , aug = params.train_aug )
         
        test_few_shot_params    = dict(n_way = params.test_n_way, n_support = params.n_shot)
        val_datamgr             = SetDataManager(params.dataset,image_size, n_query = n_query, **test_few_shot_params)
        val_loader              = val_datamgr.get_data_loader( val_file, aug = False) 
                

        if params.method == 'cad':
            model = CAD(params, model_dict[params.model], **train_few_shot_params)
        elif params.method == 'proto':
            model = ProtoNet(model_dict[params.model], **train_few_shot_params)
        else:
            raise ValueError('Unknown method')
    else:
       raise ValueError('Unknown method')

    model = model.cuda()
    print()
    print_network(model, True)
    print()
    print(f'DATASET {params.dataset}')
    print(f'METHOD  {params.method}')

    

    params.checkpoint_dir = '%s/checkpoints/%s/%s_%s' %(configs.save_dir, params.dataset, params.model, params.method)
    if params.train_aug:
        params.checkpoint_dir += '_aug'
    if not params.method  in ['baseline', 'baseline++']: 
        params.checkpoint_dir += '_%dway_%dshot' %( params.train_n_way, params.n_shot)

    if not os.path.isdir(params.checkpoint_dir):
        os.makedirs(params.checkpoint_dir)

    start_epoch = params.start_epoch
    stop_epoch  = params.stop_epoch

    if params.resume:
        resume_file = get_resume_file(params.checkpoint_dir)
        if resume_file is not None:
            tmp = torch.load(resume_file)
            start_epoch = tmp['epoch']+1
            model.load_state_dict(tmp['state'])
    elif params.warmup: #We also support warmup from pretrained baseline feature, but we never used in our paper
        baseline_checkpoint_dir = '%s/checkpoints/%s/%s_%s' %(configs.save_dir, params.dataset, params.model, 'baseline')
        if params.train_aug:
            baseline_checkpoint_dir += '_aug'
        warmup_resume_file = get_resume_file(baseline_checkpoint_dir)
        tmp = torch.load(warmup_resume_file)
        if tmp is not None: 
            state = tmp['state']
            state_keys = list(state.keys())
            for i, key in enumerate(state_keys):
                if "feature." in key:
                    newkey = key.replace("feature.","")  # an architecture model has attribute 'feature', load architecture feature to backbone by casting name from 'feature.trunk.xx' to 'trunk.xx'  
                    state[newkey] = state.pop(key)
                else:
                    state.pop(key)
            model.feature.load_state_dict(state)
        else:
            raise ValueError('No warm_up file')

    model = train(base_loader, val_loader,  model, params.optim, start_epoch, stop_epoch, params)
