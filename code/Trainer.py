import copy
import os
import time

import tqdm
from tqdm import tqdm
from metrics import *
from zmq import device
import torch
# from .layers import *
from torch import nn

def my_cos(representation1, representation2):
    sim = (1 + ((torch.sum(representation1 * representation2, 1) / (
            torch.sqrt(torch.sum(torch.pow(representation1, 2), 1)) * torch.sqrt(
        torch.sum(torch.pow(representation2, 2), 1))) + 1e-8))) / 2
    return sim

class ContrastiveLoss(nn.Module):
    def __init__(self, margin=2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1,output2,label1,label2,loss):
        # 正对or负对？
        label = torch.abs(torch.abs(label1 - label2)-1).float()
        # euclidean_distance = torch.cosine_similarity(output1, output2)
        euclidean_distance = my_cos(output1,output2)
        loss_contrastive = loss(euclidean_distance,label)
        return loss_contrastive

# input1 = torch.rand(16,1024)
# input2 = torch.rand(16,1024)
# label = torch.rand(16,1)
# euclidean_distance = torch.cosine_similarity(input1, input2)
# print(euclidean_distance.shape)

class Trainer3():
    def __init__(self,
                model,
                 contrastive_model,
                 device,
                 lr,
                 dropout,
                 dataloaders,
                 # contrastive_train_dataloader,
                 weight_decay,
                 save_param_path,
                 writer, 
                 epoch_stop,
                 epoches,
                 mode,
                 model_name, 
                 event_num,
                 save_threshold = 0.0, 
                 start_epoch = 0,
                 ):
        
        self.model = model
        self.contrastive_model = contrastive_model.cuda()
        self.device = device
        self.mode = mode
        self.model_name = model_name
        self.event_num = event_num

        self.dataloaders = dataloaders
        # self.contrastive_train_dataloader = contrastive_train_dataloader
        self.start_epoch = start_epoch
        self.num_epochs = epoches
        self.epoch_stop = epoch_stop
        self.save_threshold = save_threshold
        self.writer = writer

        if os.path.exists(save_param_path):
            self.save_param_path = save_param_path
        else:
            self.save_param_path = os.makedirs(save_param_path)
            self.save_param_path= save_param_path

        self.lr = lr
        self.weight_decay = weight_decay
        self.dropout = dropout
    
        self.criterion = nn.CrossEntropyLoss()
        # self.loss_function_auxiliary = nn.CosineEmbeddingLoss(reduction='mean')
        self.loss_function_auxiliary = ContrastiveLoss()


    def train(self):
        since = time.time()
        self.model.cuda()
        best_model_wts_val = copy.deepcopy(self.model.state_dict())
        best_acc_val = 0.0
        best_epoch_val = 0
        loss_auxiliary_total = []
        is_earlystop = False
        # 第一阶段训练：对比学习
        # Contrastive_epoch = 100
        # self.contrastive_optimizer = torch.optim.Adam(params=self.contrastive_model.parameters(), lr=0.0001)
        # for epoch in range(Contrastive_epoch):
        #     for batch in tqdm(self.contrastive_train_dataloader):
        #         batch_data = batch
        #         # 把每个样本都放到gpu上
        #         for k, v in batch_data.items():
        #             batch_data[k] = v.cuda()
        #         label = batch_data['labels']
        #
        #         with torch.set_grad_enabled(True):
        #             output = self.contrastive_model(**batch_data)
        #             # 切分一个batch的样本，形成正负例  output b(256)*dim  label [b,]
        #             output1 = output[:128,:]  #output1 b/2(128)*dim
        #             lahbels1 = label[:128]     #labels1 [b/2,]
        #             output2 = output[128:, :]
        #             lahbels2 = label[128:]
        #             loss_auxiliary = self.loss_function_auxiliary(output1, output2, lahbels1,lahbels2,self.criterion)
        #             self.contrastive_optimizer.zero_grad()
        #             loss_auxiliary.backward()
        #             self.contrastive_optimizer.step()
        #             loss_auxiliary_total.append(loss_auxiliary.item())
        #     print(f"Epoch {epoch + 1}/{epoch}, Loss: {np.mean(np.array(loss_auxiliary_total)):.4f}")
        #     # self.contrastive_model.eval()
        # torch.save(self.contrastive_model.state_dict(), 'contrastive_model.pth')
        # self.contrastive_model.eval()

        # 读取第一阶段的预训练模型
        self.contrastive_model.load_state_dict(torch.load('checkpoints/contrastive_model.pth'))
        self.contrastive_model.eval()
        for epoch in range(self.start_epoch, self.start_epoch+self.num_epochs):
            if is_earlystop:
                break
            print('-' * 50)
            print('Epoch {}/{}'.format(epoch+1, self.start_epoch+self.num_epochs))
            print('-' * 50)

            # 更新学习率
            p = float(epoch) / 100
            lr = self.lr / (1. + 10 * p) ** 0.75
            # 创建优化器
            self.optimizer = torch.optim.Adam(params=self.model.parameters(), lr=lr)
            
            for phase in ['train', 'val', 'test']:
                if phase == 'train':
                    self.model.train()  
                else:
                    self.model.eval()   
                print('-' * 10)
                print (phase.upper())
                print('-' * 10)

                running_loss_fnd = 0.0
                running_loss = 0.0 
                tpred = []
                tlabel = []

                for batch in tqdm(self.dataloaders[phase]):
                    batch_data=batch
                    # 把每个样本都放到gpu上
                    for k,v in batch_data.items():
                        batch_data[k]=v.cuda()
                    label = batch_data['label']

                    self.optimizer.zero_grad()

                    with torch.set_grad_enabled(phase == 'train'):
                        comments_fea = self.contrastive_model(batch_data['comments_fea'])
                        outputs = self.model(comments_fea,**batch_data)
                        _, preds = torch.max(outputs, 1)
                        loss = self.criterion(outputs, label)
                        if phase == 'train':
                            loss.backward()
                            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                            self.optimizer.step()
                            self.optimizer.zero_grad()

                    tlabel.extend(label.detach().cpu().numpy().tolist())
                    tpred.extend(preds.detach().cpu().numpy().tolist())
                    running_loss += loss.item() * label.size(0)
                    
                epoch_loss = running_loss / len(self.dataloaders[phase].dataset)
                print('Loss: {:.4f} '.format(epoch_loss))
                results = metrics(tlabel, tpred)
                print (results)
                self.writer.add_scalar('Loss/'+phase, epoch_loss, epoch+1)
                self.writer.add_scalar('Acc/'+phase, results['acc'], epoch+1)
                self.writer.add_scalar('F1/'+phase, results['f1'], epoch+1)
                if phase == 'val' and results['acc'] > best_acc_val:
                    best_acc_val = results['acc']
                    best_epoch_val = epoch+1
                    if best_acc_val > self.save_threshold:
                        torch.save(self.model.state_dict(), self.save_param_path + "_val_epoch" + str(best_epoch_val) + "_{0:.4f}".format(best_acc_val))
                        print('---------------------------------------------------------')
                        print ("saved " + self.save_param_path + "_test_epoch" + str(best_epoch_val) + "_{0:.4f}".format(best_acc_val) )
                        print(results)
                        print('---------------------------------------------------------')
                    # else:
                    #     if epoch-best_epoch_val >= self.epoch_stop-1:
                    #         is_earlystop = True
                    #         print ("early stopping...")

        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60))
        print("Best model on val: epoch" + str(best_epoch_val) + "_" + str(best_acc_val))

        # self.model.load_state_dict(best_model_wts_val)

        print ("test result when using best model on val")
        return True

class Trainer_event():
    def __init__(self,
                 model,
                 contrastive_model,
                 device,
                 lr,
                 dropout,
                 dataloaders,
                 weight_decay,
                 save_param_path,
                 writer,
                 epoch_stop,
                 epoches,
                 mode,
                 model_name,
                 event_num,
                 save_threshold=0.0,
                 start_epoch=0,
                 ):

        self.model = model
        self.contrastive_model = contrastive_model.cuda()
        self.device = device
        self.mode = mode
        self.model_name = model_name
        self.event_num = event_num

        self.dataloaders = dataloaders
        self.start_epoch = start_epoch
        self.num_epochs = epoches
        self.epoch_stop = epoch_stop
        self.save_threshold = save_threshold
        self.writer = writer

        if os.path.exists(save_param_path):
            self.save_param_path = save_param_path
        else:
            self.save_param_path = os.makedirs(save_param_path)
            self.save_param_path = save_param_path

        self.lr = lr
        self.weight_decay = weight_decay
        self.dropout = dropout

        self.criterion = nn.CrossEntropyLoss()

        # 读取第一阶段的预训练模型
        self.contrastive_model.load_state_dict(torch.load('checkpoints/contrastive_model.pth'))
        self.contrastive_model.eval()

    def train(self):
        since = time.time()
        self.model.cuda()
        best_model_wts_val = copy.deepcopy(self.model.state_dict())
        best_acc_val = 0.0
        best_epoch_val = 0

        is_earlystop = False

        for epoch in range(self.start_epoch, self.start_epoch + self.num_epochs):
            if is_earlystop:
                break
            print('-' * 50)
            print('Epoch {}/{}'.format(epoch + 1, self.start_epoch + self.num_epochs))
            print('-' * 50)

            # 更新学习率
            # p = float(epoch) / 100
            # lr = self.lr / (1. + 10 * p) ** 0.75
            lr = 0.0005
            # 创建优化器
            self.optimizer = torch.optim.Adam(params=self.model.parameters(), lr=lr)

            for phase in ['train','test']:
                if phase == 'train':
                    self.model.train()
                else:
                    self.model.eval()
                # self.model.train()
                print('-' * 10)
                print(phase.upper())
                print('-' * 10)

                running_loss_fnd = 0.0
                running_loss = 0.0
                tpred = []
                tlabel = []

                for batch in tqdm(self.dataloaders[phase]):
                    batch_data = batch
                    # 把每个样本都放到gpu上
                    for k, v in batch_data.items():
                        batch_data[k] = v.cuda()
                    label = batch_data['label']

                    self.optimizer.zero_grad()
                    with torch.set_grad_enabled(phase == 'train'):
                        comments_fea = self.contrastive_model(batch_data['comments_fea'])
                        outputs = self.model(comments_fea,**batch_data)
                        _, preds = torch.max(outputs, 1)
                        loss = self.criterion(outputs, label)
                        if phase == 'train':
                            loss.backward()
                            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                            self.optimizer.step()
                            self.optimizer.zero_grad()

                    tlabel.extend(label.detach().cpu().numpy().tolist())
                    tpred.extend(preds.detach().cpu().numpy().tolist())
                    running_loss += loss.item() * label.size(0)

                epoch_loss = running_loss / len(self.dataloaders[phase].dataset)
                print('Loss: {:.4f} '.format(epoch_loss))
                results = metrics(tlabel, tpred)
                print(results)
                self.writer.add_scalar('Loss/' + phase, epoch_loss, epoch + 1)
                self.writer.add_scalar('Acc/' + phase, results['acc'], epoch + 1)
                self.writer.add_scalar('F1/' + phase, results['f1'], epoch + 1)

                if phase == 'test' and results['acc'] > best_acc_val:
                    best_acc_val = results['acc']
                    best_model_wts_val = copy.deepcopy(self.model.state_dict())
                    best_epoch_val = epoch + 1
                    if best_acc_val > self.save_threshold:
                        torch.save(self.model.state_dict(),
                                   self.save_param_path + "_test_epoch" + str(best_epoch_val) + "_{0:.4f}".format(
                                       best_acc_val))
                        print("saved " + self.save_param_path + "_test_epoch" + str(
                            best_epoch_val) + "_{0:.4f}".format(best_acc_val))
                    else:
                        if epoch - best_epoch_val >= self.epoch_stop - 1:
                            is_earlystop = True
                            print("early stopping...")

        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60))
        print("Best model on val: epoch" + str(best_epoch_val) + "_" + str(best_acc_val))
        return True
    
