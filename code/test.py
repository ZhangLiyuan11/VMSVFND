import torch
from tqdm import tqdm

from VMSVFND import VEMFSVModel,Contrastive_learning_model
from dataloader import VMSVFNDDataset, collate_fn
from torch.utils.data import DataLoader
from metrics import *

contrastive_model_path = 'checkpoints/contrastive_model.pth'
model_path = 'checkpoints/test_0.8672'

def load_chechpoint(contrastive_model_path,model_path):
    model = VEMFSVModel(dropout=0.1, fea_dim=128)
    model.load_state_dict(torch.load(model_path),strict=False)
    model.eval()
    contrastive_model = Contrastive_learning_model()
    contrastive_model.load_state_dict(torch.load(contrastive_model_path))
    contrastive_model.eval()
    return model.cuda(),contrastive_model.cuda()

def get_dataloader():
    dataset_test = VMSVFNDDataset('vid_time3_test.txt')
    test_dataloader = DataLoader(dataset_test, batch_size=16,
                                 num_workers=0,
                                 pin_memory=True,
                                 shuffle=False,
                                 collate_fn=collate_fn)
    return test_dataloader

def test():
    model,contrastive_model = load_chechpoint(contrastive_model_path,model_path)
    test_dataloader = get_dataloader()
    tpred = []
    tlabel = []
    for batch in tqdm(test_dataloader):
        batch_data = batch
        # 把每个样本都放到gpu上
        for k, v in batch_data.items():
            batch_data[k] = v.cuda()
        label = batch_data['label']
        with torch.set_grad_enabled(False):
            comments_fea = contrastive_model(batch_data['comments_fea'])
            outputs = model(comments_fea,**batch_data)
            _,preds = torch.max(outputs, 1)
        tlabel.extend(label.detach().cpu().numpy().tolist())
        tpred.extend(preds.detach().cpu().numpy().tolist())
    results = metrics(tlabel, tpred)
    print(results)


if __name__ == '__main__':
    test()