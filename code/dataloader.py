import os
import pickle

import h5py
import numpy as np
import pandas as pd
import torch

from torch.utils.data import Dataset
from tools import *

def str2num(str_x):
    # isinstance判断一个对象是不是一个已知类型
    if isinstance(str_x, float):
        return str_x
    # isdigit 检测字符串是否只由数字组成
    elif str_x.isdigit():
        return int(str_x)
    elif 'w' in str_x:
        return float(str_x[:-1])*10000
    elif '亿' in str_x:
        return float(str_x[:-1])*100000000
    else:
        print ("error")
        print (str_x)

class ContrastiveDataset(Dataset):
    def __init__(self, comments, comments_likes, labels):
        self.comments = comments
        self.comments_likes = comments_likes
        self.labels = labels

    def __len__(self):
        return len(self.comments)

    def __getitem__(self, item):
        comments_feas = torch.squeeze(torch.stack(self.comments[item]),dim=1)
        comments_likes = self.comments_likes[item]
        # comments_like = []
        # for num in comments_likes:
        #     num_like = num.split(" ")[0]
        #     comments_like.append(str2num(num_like))
        comments_likes = torch.tensor(comments_likes)
        # torch.true_divide 返回除法的浮点数
        comments_weight = torch.stack(
            [torch.true_divide((i + 1), (comments_likes.shape[0] + comments_likes.sum())) for i in
             comments_likes])
        comments_fea = torch.sum(comments_feas * (comments_weight.reshape(comments_weight.shape[0], 1)),dim=0)

        return {
            'comments_feas': comments_fea,
            'comments_likes': comments_likes,
            'label': self.labels[item],
        }

def collate_fn_contrastive(batch):
    comments_fea = torch.stack([item['comments_feas'] for item in batch])
    # comments_like = [item['comments_likes'] for item in batch]
    labels = torch.tensor([item['label'] for item in batch])

    return {
        'comments_fea': comments_fea,
        # 'comments_like': comments_like,
        'labels': labels,
    }

class VMSVFNDDataset(Dataset):
    def __init__(self, path_vid, datamode='title+ocr'):
        
        # with open(r'../data/dict_vid_audioconvfea.pkl', "rb") as fr:
        with open(r'..\dataset\data\dict_vid_audioconvfea.pkl', "rb") as fr:
            self.dict_vid_convfea = pickle.load(fr)
        self.data_complete = pd.read_json('..\dataset\data\data_complete.json', orient='records', dtype=False, lines=True)

        # self.data_complete = self.data_complete[self.data_complete['label']!=2] # label: 0-real, 1-fake, 2-debunk
        self.data_complete = self.data_complete[self.data_complete['annotation'] != '辟谣']
        self.framefeapath='..\dataset\data\ptvgg19_frames'
        self.c3dfeapath='..\dataset\data\c3d\\'
        self.c3d_noise_feapath = '..\dataset\data\c3d_SRM\\'
        self.comments_fea_path = '..\dataset\data\comments_embed\\'

        self.vid = []
        
        with open('..\dataset\\vids\\'+path_vid, "r") as fr:
            for line in fr.readlines():
                self.vid.append(line.strip())

        self.data = self.data_complete[self.data_complete.video_id.isin(self.vid)]
        # 设置类标签
        self.data['video_id'] = self.data['video_id'].astype('category')
        # 改变标签类别
        self.data['video_id'].cat.set_categories(self.vid)
        # 按照类别排序，而不是字母排序
        self.data.sort_values('video_id', ascending=True, inplace=True)
        # 重置数据帧的索引，并使用默认索引
        self.data.reset_index(inplace=True)  

        self.tokenizer = pretrain_bert_token()
        # self.tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')

        self.datamode = datamode
        
    def __len__(self):
        return self.data.shape[0]
     
    def __getitem__(self, idx):
        item = self.data.iloc[idx]
        vid = item['video_id']

        # label 
        label = 0 if item['annotation']=='真' else 1
        label = torch.tensor(label)

        # text
        if self.datamode == 'title+ocr':
            title_tokens = self.tokenizer(item['title']+' '+item['ocr']+''+item['keywords'], max_length=512, padding='max_length', truncation=True)
        elif self.datamode == 'ocr':
            title_tokens = self.tokenizer(item['ocr']+''+item['keywords'], max_length=512, padding='max_length', truncation=True)
        elif self.datamode == 'title':
            title_tokens = self.tokenizer(item['title']+''+item['keywords'], max_length=512, padding='max_length', truncation=True)
        title_inputid = torch.LongTensor(title_tokens['input_ids'])
        title_mask = torch.LongTensor(title_tokens['attention_mask'])

        # comments 观点-情感
        try:
            comments = torch.load(self.comments_fea_path + vid + '.pkl')
            comments_fea = torch.squeeze(torch.stack(comments['comments_em']),dim=1)
            comments_like = torch.tensor(comments['comments_like'])
            comments_weight = torch.stack(
            [torch.true_divide((i + 1), (comments_like.shape[0] + comments_like.sum())) for i in
             comments_like])
            comments_fea = torch.unsqueeze(torch.sum(comments_fea * (comments_weight.reshape(comments_weight.shape[0], 1)),
                                          dim=0),dim=0)
        except:
            comments_fea = torch.zeros(1,1024)
            comments_like = []

        # audio
        audioframes = self.dict_vid_convfea[vid]
        audioframes = torch.FloatTensor(audioframes)

        # path = os.path.join(self.framefeapath,vid+'.pkl')
        # frames
        frames=pickle.load(open(os.path.join(self.framefeapath,vid+'.pkl'),'rb'))
        frames=torch.FloatTensor(frames)

        # path = os.path.join(self.c3dfeapath+vid+".hdf5")
        # video
        c3d = h5py.File(self.c3dfeapath+vid+".hdf5", "r")[vid]['c3d_features']
        c3d = torch.FloatTensor(c3d)

        # mask_video
        noise_c3d = h5py.File(self.c3d_noise_feapath+vid+".hdf5", "r")[vid]['c3d_features']
        noise_c3d = torch.FloatTensor(noise_c3d)
        # # user
        try: 
            if item['is_author_verified'] == 1:
                intro = "个人认证"
            elif item['is_author_verified'] == 2:
                intro = "机构认证"
            elif item['is_author_verified'] == 0:
                intro = "未认证"
            else: 
                intro = "认证状态未知"
        except:
            if 'author_verified_intro' == '':
                intro = "认证状态未知"
            else:
                intro = "有认证"

        for key in ['author_intro', 'author_verified_intro']:
            try:
                intro = intro + ' ' + item[key]
            except:
                intro += ' '

        intro_tokens = self.tokenizer(intro, max_length=50, padding='max_length', truncation=True)
        intro_inputid = torch.LongTensor(intro_tokens['input_ids'])
        intro_mask = torch.LongTensor(intro_tokens['attention_mask'])

        return {
            'label': label,
            'title_inputid': title_inputid,
            'title_mask': title_mask,
            'audioframes': audioframes,
            'frames':frames,
            'c3d': c3d,
            'noise_c3d':noise_c3d,
            'comments_fea': comments_fea,
            # 'comments_like': comments_like,
            # 'comments_inputid':comments_inputid,
            # 'comments_mask':comments_mask,
            # 'comments_like_raw':comments_like_raw,
            'intro_inputid': intro_inputid,
            'intro_mask': intro_mask,
        }



def pad_sequence(seq_len,lst, emb):
    result=[]
    for video in lst:
        # isinstance()函数来判断一个对象是否是一个已知的类型
        if isinstance(video, list):
            if len(video) == 0:
                result.append(torch.zeros([seq_len,emb],dtype=torch.float))
                continue
            video = torch.stack(video)
            video = torch.squeeze(video,dim=1)
        ori_len=video.shape[0]
        if ori_len == 0:
            video = torch.zeros([seq_len,emb],dtype=torch.float)
        elif ori_len >= seq_len:
            if emb == 200:
                video=torch.FloatTensor(video[:seq_len])
            else:
                video=torch.FloatTensor(video[:seq_len])
        else:
            video=torch.cat([video,torch.zeros([seq_len-ori_len,video.shape[1]],dtype=torch.float)],dim=0)
            if emb == 200:
                video=torch.FloatTensor(video)
            else:
                video=torch.FloatTensor(video)
        result.append(video)
    return torch.stack(result)

def pad_sequence2(seq_len,lst, emb):
    result=[]
    for video in lst:
        # isinstance()函数来判断一个对象是否是一个已知的类型
        if isinstance(video, list):
            if len(video) == 0:
                result.append(torch.zeros([seq_len,emb],dtype=torch.long))
                continue
            video = torch.stack(video)
            video = torch.squeeze(video,dim=1)
        ori_len=video.shape[0]
        if ori_len == 0:
            video = torch.zeros([seq_len,emb],dtype=torch.long)
        elif ori_len >= seq_len:
            if emb == 200:
                video=torch.FloatTensor(video[:seq_len])
            else:
                video=torch.LongTensor(video[:seq_len])
        else:
            video=torch.cat([video,torch.zeros([seq_len-ori_len,video.shape[1]],dtype=torch.long)],dim=0)
            if emb == 200:
                video=torch.FloatTensor(video)
            else:
                video=torch.LongTensor(video)
        result.append(video)
    return torch.stack(result)

def pad_frame_sequence(seq_len,lst):
    attention_masks = []
    result=[]
    for video in lst:
        video=torch.FloatTensor(video)
        ori_len=video.shape[0]
        if ori_len>=seq_len:
            gap=ori_len//seq_len
            video=video[::gap][:seq_len]
            mask = np.ones((seq_len))
        else:
            video=torch.cat((video,torch.zeros([seq_len-ori_len,video.shape[1]],dtype=torch.float)),dim=0)
            mask = np.append(np.ones(ori_len), np.zeros(seq_len-ori_len))
        result.append(video)
        mask = torch.IntTensor(mask)
        attention_masks.append(mask)
    return torch.stack(result), torch.stack(attention_masks)

def collate_fn(batch):
    num_comments = 23
    num_frames = 83
    num_audioframes = 50

    intro_inputid = [item['intro_inputid'] for item in batch]
    intro_mask = [item['intro_mask'] for item in batch]

    title_inputid = [item['title_inputid'] for item in batch]
    title_mask = [item['title_mask'] for item in batch]

    # batch_comments_like = [item['comments_like'] for item in batch]
    comments_fea = [item['comments_fea'] for item in batch]


    # 对评论数据进行pad操作补齐为同一维度
    # comments_fea = pad_sequence(num_comments, comments_fea, 1024)

    # 根据帧数补齐关键帧特征
    frames = [item['frames'] for item in batch]
    frames, frames_masks = pad_frame_sequence(num_frames, frames)

    # 根据语音帧数补齐语音特征
    audioframes = [item['audioframes'] for item in batch]
    audioframes, audioframes_masks = pad_frame_sequence(num_audioframes, audioframes)

    # 补齐C3D特征
    c3d = [item['c3d'] for item in batch]
    c3d, c3d_masks = pad_frame_sequence(num_frames, c3d)

    noise_c3d = [item['noise_c3d'] for item in batch]
    noise_c3d, noise_c3d_mask = pad_frame_sequence(num_frames, noise_c3d)

    label = [item['label'] for item in batch]

    return {
        'label': torch.stack(label),
        'intro_inputid': torch.stack(intro_inputid),
        'intro_mask': torch.stack(intro_mask),
        'title_inputid': torch.stack(title_inputid),
        'title_mask': torch.stack(title_mask),
        'comments_fea': torch.stack(comments_fea),
        'audioframes': audioframes,
        'audioframes_masks': audioframes_masks,
        'frames': frames,
        'frames_masks': frames_masks,
        'c3d': c3d,
        'noise_c3d':noise_c3d,
        'c3d_masks': c3d_masks,
    }

