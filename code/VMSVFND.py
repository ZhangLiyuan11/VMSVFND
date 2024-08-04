from transformers import BertModel
import torch
from torch import nn

from trans_model import Transformer
from LSTM_layer import *
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
from tools import *
from CBP_layer import *
class VEMFSVModel(torch.nn.Module):
    def __init__(self,fea_dim,dropout):
        super(VEMFSVModel, self).__init__()
        # 加载bert模型
        self.bert = pretrain_bert_models()

        # 维度
        self.text_dim = 1024
        self.comment_dim = 1024
        self.img_dim = 4096
        self.video_dim = 4096
        self.num_frames = 83
        self.num_audioframes = 50
        self.num_comments = 23
        # fea_dim 128
        self.dim = fea_dim
        self.num_heads = 4

        self.trans_dim = 128

        self.dropout = dropout
        # 预训练的vggish模型处理音频
        self.vggish_layer = torch.hub.load('harritaylor/torchvggish', 'vggish', source='github')
        net_structure = list(self.vggish_layer.children())
        self.vggish_modified = nn.Sequential(*net_structure[-2:-1])
        # # 语义交互共注意力模块
        self.at_CT = Transformer(model_dimension=self.trans_dim, number_of_heads=self.num_heads, dropout_probability=self.dropout)
        self.tv_CT = Transformer(model_dimension=self.trans_dim, number_of_heads=self.num_heads, dropout_probability=self.dropout)
        self.vt_CT = Transformer(model_dimension=self.trans_dim, number_of_heads=self.num_heads, dropout_probability=self.dropout)
        #
        self.CBP_layer = CompactBilinearPooling(128, 128, 128)


        self.linear_text = nn.Sequential(torch.nn.Linear(self.text_dim, fea_dim), torch.nn.ReLU(),nn.Dropout(p=self.dropout))
        self.linear_comment = nn.Sequential(torch.nn.Linear(fea_dim, fea_dim), torch.nn.ReLU(),nn.Dropout(p=self.dropout))
        self.linear_img = nn.Sequential(torch.nn.Linear(self.img_dim, fea_dim), torch.nn.ReLU(),nn.Dropout(p=self.dropout))
        self.linear_video = nn.Sequential(torch.nn.Linear(self.video_dim, fea_dim), torch.nn.ReLU(),nn.Dropout(p=self.dropout))
        self.linear_video_noise = nn.Sequential(torch.nn.Linear(self.video_dim, fea_dim), torch.nn.ReLU(),nn.Dropout(p=self.dropout))
        self.linear_intro = nn.Sequential(torch.nn.Linear(self.text_dim, fea_dim),torch.nn.ReLU(),nn.Dropout(p=self.dropout))
        self.linear_audio = nn.Sequential(torch.nn.Linear(fea_dim, fea_dim), torch.nn.ReLU(),nn.Dropout(p=self.dropout))
        self.linear_cbp = nn.Sequential(torch.nn.Linear(fea_dim, fea_dim), torch.nn.ReLU(),nn.Dropout(p=self.dropout))
        # 最终的融合模块
        self.mix_layer = mix_fea_layer(fea_dim)

        self.classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(64, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 2)
        )

    def forward(self, comments, **kwargs):
        ### User Intro ###
        intro_inputid = kwargs['intro_inputid']
        intro_mask = kwargs['intro_mask']
        fea_intro = self.bert(intro_inputid,attention_mask=intro_mask)[1]
        fea_intro = self.linear_intro(fea_intro)

        ### Title ###
        title_inputid = kwargs['title_inputid']#(batch,512)
        title_mask=kwargs['title_mask']#(batch,512)
        fea_text=self.bert(title_inputid,attention_mask=title_mask)['last_hidden_state']#(batch,sequence,768)
        fea_text=self.linear_text(fea_text)

        ### Audio Frames ###
        audioframes=kwargs['audioframes']#(batch,36,12288)
        fea_audio = self.vggish_modified(audioframes) #(batch, frames, 128)
        fea_audio = self.linear_audio(fea_audio)

        ### Image Frames ###
        frames=kwargs['frames']#(batch,30,4096)
        fea_img = self.linear_img(frames)

        # 语义交互
        fea_at = self.at_CT(fea_audio, fea_text)
        fea_at = torch.mean(fea_at, -2)
        fea_tv = self.tv_CT(fea_text,fea_img)
        fea_tv = torch.mean(fea_tv, -2)
        fea_vt = self.vt_CT(fea_img,fea_text)
        fea_vt = torch.mean(fea_vt, -2)
        #
        # ### C3D ###
        c3d = kwargs['c3d'] # (batch, 36, 4096)
        fea_video_c3d = self.linear_video(c3d) #(batch, frames, 128)

        noise_c3d = kwargs['noise_c3d']  # (batch, 36, 4096)
        fea_video_noise = self.linear_video_noise(noise_c3d)  # (batch, frames, 128)

        fea_video = self.CBP_layer(fea_video_c3d, fea_video_noise)
        fea_video = self.linear_cbp(fea_video)

        # # 特征融合
        comments = self.linear_comment(comments)
        fea = self.mix_layer(fea_vt,fea_tv,fea_at,fea_video,comments,fea_intro)
        # fea = self.mix_layer(fea_video, fea_vt, fea_tv, fea_at, comments)
        output = self.classifier(fea)

        return output


class Contrastive_learning_model(nn.Module):
    def __init__(self):
        super(Contrastive_learning_model, self).__init__()
        # 方案1 使用共享参数的映射头
        # 方案2 一个辅助的映射头，只过正例
        self.uni_net_share = nn.Sequential(
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ELU(),
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.ELU(),
            nn.Linear(128, 128),
            nn.BatchNorm1d(128),
            nn.ELU(),
        )
        # self.pic_alignment = nn.Sequential(
        #     nn.Linear(1024, 512),
        #     nn.BatchNorm1d(512),
        #     nn.ELU(),
        #     nn.Linear(512, 128),
        #     nn.BatchNorm1d(128),
        #     nn.ELU(),
        #     nn.Linear(128, 64),
        #     nn.BatchNorm1d(64),
        #     nn.ELU(),
        # )

    def forward(self, comments_fea):
        out = self.uni_net_share(torch.squeeze(comments_fea))
        return out