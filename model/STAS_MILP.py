import torch
from torch import nn
from torch_geometric.utils import softmax
from torch_scatter import scatter
from club import utils


class Attn_Net_Gated(nn.Module):
    def __init__(self, D, L=128, temp=1., dropout=0.35,scale=0.8,th=0.8,add_margin=False):
        super(Attn_Net_Gated, self).__init__()
        self.scale=scale
        self.th=th
        self.add_margin=add_margin

        self.a_attention = nn.Sequential(nn.Linear(D, L), nn.Sigmoid())
        self.b_attention = nn.Sequential(nn.Linear(D, L), nn.Tanh())
        self.linear = nn.Sequential(nn.Linear(L, 1))

        self.dropout = nn.Dropout(dropout, self.training)
        self.t = temp
        self.init_weight()

    def init_weight(self):
        for a_fc, b_fc, c_fc in zip(self.a_attention, self.b_attention, self.linear):
            if isinstance(a_fc, nn.Linear):
                nn.init.xavier_normal_(a_fc.weight)
            if isinstance(b_fc, nn.Linear):
                nn.init.xavier_normal_(b_fc.weight)
            if isinstance(c_fc, nn.Linear):
                nn.init.xavier_normal_(c_fc.weight)

    def add_margin_2_attn(self,score,batch):
        margin_score=[]
        for b in range(batch.max()+1):
            current_socre=score[batch==b,:]
            attn_normalized=utils.Max_MIN_Tensor(current_socre)
            mask = torch.where(
                attn_normalized>self.th,
                torch.tensor(self.scale),
                torch.tensor(1.0)
            )
            attn_scores = current_socre * mask
            margin_score.append(attn_scores)
        return torch.cat(margin_score)

    def forward(self, feature, batch, istrain=True):
        a = self.a_attention(feature)
        b = self.b_attention(feature)
        a = a * b
        score = self.linear(a)

        if self.add_margin:
            score=self.add_margin_2_attn(score,batch)

        score_softmax = softmax(score, batch)

        # score = score / self.t

        # if istrain:
        #     score_softmax = self.dropout(score_softmax)
        out = scatter(score_softmax * feature, batch, 0)
        return out, score, score_softmax, feature


class STAS_MIL(nn.Module):
    def __init__(self, n_class=2,
                 input_dim=768,
                 embed_dim=256,
                 AL=128,
                 AD=0.35,
                 AT=1,
                 scale=0.9,
                 th=0.8,
                 add_margin=False,
                ):
        super(STAS_MIL, self).__init__()
        self.n_class = n_class
        self.input_dim = input_dim
        self.embed_dim = embed_dim

        self.fc = nn.Sequential(
            nn.BatchNorm1d(self.input_dim),
            nn.Linear(self.input_dim, self.embed_dim),
            nn.ReLU(inplace=True),
        )

        # 注意力机制
        self.Attention = Attn_Net_Gated(
            D=self.embed_dim,
            dropout=AD,
            L=AL,
            temp=AT,
            scale=scale,
            th=th,
            add_margin=add_margin
        )
        self.classifier = nn.Sequential(nn.Linear(self.embed_dim, n_class),
                                       nn.Sigmoid())

        self.init_weight()

    def init_weight(self):
        for m in self.fc.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight, gain=1)
        # nn.init.xavier_normal_(self.classifier, gain=1)
        self.Attention.init_weight()

    def forward(self,data):
        x=data["x"]
        batch=data["batch"]

        x = self.fc(x)
        x_norm, score, score_softmax, patch_norm_feature = self.Attention(x, batch, True)

        probs = self.classifier(x_norm)

        result={}
        result['score']=score
        result['score_softmax']=score_softmax
        result['patch_norm_feature']=patch_norm_feature
        result['probs']=probs
        return result

