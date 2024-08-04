from torch import nn
import torch

class myGruLayer(torch.nn.Module):
    def __init__(self,  size_out=128):
        super(myGruLayer, self).__init__()
        # self.size_in1, self.size_in2, self.size_out = 128, 128, size_out

        self.hidden1 = nn.Linear(128, size_out, bias=False)
        self.hidden2 = nn.Linear(128, size_out, bias=False)
        self.hidden_sigmoid = nn.Linear(256, 1, bias=False)

        # Activation functions
        self.tanh_f = nn.Tanh()
        self.sigmoid_f = nn.Sigmoid()

    def forward(self, x1, x2):
        h1 = self.tanh_f(self.hidden1(x1))
        h2 = self.tanh_f(self.hidden1(x2))
        x = torch.cat((h1, h2), dim=1)
        z = self.sigmoid_f(self.hidden_sigmoid(x))

        return z.view(z.size()[0], 1) * x1 + (1 - z).view(z.size()[0], 1) * x2, z




class mix_fea_layer(torch.nn.Module):
    def __init__(self, fea_dim):
        super(mix_fea_layer, self).__init__()

        self.fea_dim = fea_dim

        self.gru_layer1 = myGruLayer(self.fea_dim)
        self.gru_layer2 = myGruLayer(self.fea_dim)
        self.gru_layer3 = myGruLayer(self.fea_dim)
        self.gru_layer4 = myGruLayer(self.fea_dim)
        self.gru_layer5 = myGruLayer(self.fea_dim)

    def forward(self,fea_vt,fea_ta,fea_tv,fea_video,fea_comment,fea_user):
        fea_out1,score1 = self.gru_layer1(fea_tv,fea_vt)
        fea_out2,score2 = self.gru_layer2(fea_out1,fea_ta)
        fea_out3,score3 = self.gru_layer3(fea_out2,fea_video)
        fea_out4,score4 = self.gru_layer4(fea_out3,fea_comment)
        fea_out5,score5 = self.gru_layer5(fea_out4,fea_user)
        e1 = torch.exp(score1)
        e2 = torch.exp(score2)
        e3 = torch.exp(score3)
        e4 = torch.exp(score4)
        e5 = torch.exp(score5)
        e_all = e1+e2+e3+e4+e5
        a1 = e1 / e_all
        a2 = e2 / e_all
        a3 = e3 / e_all
        a4 = e4 / e_all
        a5 = e5 / e_all
        fea_out = a1 * fea_out1 + a2 * fea_out2 + a3 * fea_out3 + a4 * fea_out4 + a5 * fea_out5
        return fea_out