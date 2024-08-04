
import collections

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from VMSVFND import VEMFSVModel,Contrastive_learning_model
from dataloader import *
from Trainer import Trainer3,Trainer_event

def _init_fn(worker_id):
    np.random.seed(2022)

class Run():
    def __init__(self,config):
        self.model_name = config['model_name']
        self.mode_eval = config['mode_eval']
        self.fold = config['fold']
        self.data_type = 'VMSVFND'
        self.epoches = config['epoches']
        self.batch_size = config['batch_size']
        self.num_workers = config['num_workers']
        self.epoch_stop = config['epoch_stop']
        self.seed = config['seed']
        self.device = config['device']
        self.lr = config['lr']
        self.lambd=config['lambd']
        self.save_param_dir = config['path_param']
        self.path_tensorboard = config['path_tensorboard']
        self.dropout = config['dropout']
        self.weight_decay = config['weight_decay']
        self.event_num = 616 
        self.mode ='normal'


    def get_dataloader_temporal(self, data_type):
        # Contrastive_data = torch.load(Contrastive_path)
        # comments_feas = Contrastive_data['all_comments_fea']
        # comments_likes = Contrastive_data['all_comments_like']
        # labels = Contrastive_data['all_labels']
        # Contrastive_train_dataset = ContrastiveDataset(comments_feas,comments_likes,labels)
        dataset_train = VMSVFNDDataset('vid_time3_train.txt')
        dataset_val = VMSVFNDDataset('vid_time3_val.txt')
        dataset_test = VMSVFNDDataset('vid_time3_test.txt')

        # Contrastive_train_dataloader = DataLoader(Contrastive_train_dataset,batch_size=256,
        #                                           shuffle=True,num_workers=self.num_workers,drop_last=True,collate_fn=collate_fn_contrastive)

        train_dataloader = DataLoader(dataset_train, batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            shuffle=True,
            worker_init_fn=_init_fn,
            collate_fn=collate_fn)
        val_dataloader = DataLoader(dataset_val, batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            shuffle=False,
            worker_init_fn=_init_fn,
            collate_fn=collate_fn)
        test_dataloader=DataLoader(dataset_test, batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            shuffle=False,
            worker_init_fn=_init_fn,
            collate_fn=collate_fn)
 
        dataloaders =  dict(zip(['train', 'val', 'test'],[train_dataloader, val_dataloader, test_dataloader]))
        return dataloaders

    def get_dataloader_nocv(self, data_type,data_fold):
        dataset_train = VMSVFNDDataset(f'vid_fold_no_{data_fold}.txt')
        dataset_test = VMSVFNDDataset(f'vid_fold_{data_fold}.txt')

        train_dataloader = DataLoader(dataset_train, batch_size=self.batch_size,
                                      num_workers=self.num_workers,
                                      pin_memory=True,
                                      shuffle=True,
                                      worker_init_fn=_init_fn,
                                      collate_fn=collate_fn,
                                      drop_last=True)
        test_dataloader = DataLoader(dataset_test, batch_size=self.batch_size,
                                     num_workers=self.num_workers,
                                     pin_memory=True,
                                     shuffle=False,
                                     worker_init_fn=_init_fn,
                                     collate_fn=collate_fn)

        dataloaders = dict(zip(['train','test'], [train_dataloader,test_dataloader]))
        return dataloaders

    def get_model(self):
        self.Contrastive_model = Contrastive_learning_model()
        if self.model_name == 'VMSVFND':
                self.model = VEMFSVModel(fea_dim=128,dropout=self.dropout)
        return self.model,self.Contrastive_model

    def main(self):
        self.model,self.Contrastive_model = self.get_model()
        if self.mode_eval == 'temporal':
            # Contrastive_path = '../contrastive_learn_data.pt'
            dataloaders = self.get_dataloader_temporal(data_type=self.data_type)
            trainer = Trainer3(model=self.model,contrastive_model=self.Contrastive_model, device = self.device, lr = self.lr, dataloaders = dataloaders, epoches = self.epoches, dropout = self.dropout, weight_decay = self.weight_decay, mode = self.mode, model_name = self.model_name, event_num = self.event_num,
                    epoch_stop = self.epoch_stop, save_param_path = self.save_param_dir+self.data_type+"/"+self.model_name+"/", writer = SummaryWriter(self.path_tensorboard))
            result = trainer.train()
            return result
        else:
            dataloaders = self.get_dataloader_nocv(data_type=self.data_type,data_fold = self.fold)
            trainer = Trainer_event(model=self.model,contrastive_model=self.Contrastive_model, device=self.device, lr=self.lr, dataloaders=dataloaders,
                               epoches=self.epoches, dropout=self.dropout, weight_decay=self.weight_decay,
                               mode=self.mode, model_name=self.model_name, event_num=self.event_num,
                               epoch_stop=self.epoch_stop,
                               save_param_path=self.save_param_dir + self.data_type + "/" + self.model_name + "/",
                               writer=SummaryWriter(self.path_tensorboard))
            result = trainer.train()
            return result

