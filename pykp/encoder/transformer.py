"""
Implementation of "Attention is All You Need"
"""
import pdb

import torch.nn as nn
import torch
import torch.nn.functional as F
import math
from pykp.modules.multi_head_attn import MultiHeadAttention
import numpy as np



class TransformerSeq2SeqEncoderLayer(nn.Module):
    def __init__(self, d_model: int = 512, n_head: int = 8, dim_ff: int = 2048,
                 dropout: float = 0.1):
        """
        Self-Attention的Layer，
        :param int d_model: input和output的输出维度
        :param int n_head: 多少个head，每个head的维度为d_model/n_head
        :param int dim_ff: FFN的维度大小
        :param float dropout: Self-attention和FFN的dropout大小，0表示不drop
        """
        super(TransformerSeq2SeqEncoderLayer, self).__init__()
        self.d_model = d_model
        self.n_head = n_head
        self.dim_ff = dim_ff
        self.dropout = dropout

        self.self_attn = MultiHeadAttention(d_model, n_head, dropout)
        self.attn_layer_norm = nn.LayerNorm(d_model)
        self.ffn_layer_norm = nn.LayerNorm(d_model)

        self.ffn = nn.Sequential(nn.Linear(self.d_model, self.dim_ff),
                                 nn.ReLU(),
                                 nn.Dropout(dropout),
                                 nn.Linear(self.dim_ff, self.d_model),
                                 nn.Dropout(dropout))

    def forward(self, x, mask):
        """
        :param x: batch x src_seq x d_model
        :param mask: batch x src_seq，为0的地方为padding
        :return:
        """
        # attention
        residual = x
        x = self.attn_layer_norm(x)

        x, _ = self.self_attn(query=x,
                              key=x,
                              value=x,
                              key_mask=mask)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x

        # ffn
        residual = x
        x = self.ffn_layer_norm(x)
        x = self.ffn(x)
        x = residual + x

        return x


class similar_attn(nn.Module):
    def __init__(self, d_model=512, dropout=0.1):
        super().__init__()
        self.d_model=d_model
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k= nn.Linear(d_model, d_model)
        self.scale = math.sqrt(self.d_model)
        self.softmax = nn.Softmax(dim=2)
        self.dropout = dropout
        self.reset_parameters()

    def forward(self,src_embed,tlt_embed,src_mask):
        trans_tlt_embed = self.w_q(tlt_embed)
        trans_src_embed=self.w_k(src_embed)
        trans_src_embed = trans_src_embed[:, :, None, :]  # 8,277,1,512
        trans_tlt_embed = trans_tlt_embed[:, :, None, :]  # 8,17,1,512
        #
        device=src_embed.device
        trans_tlt_embed = torch.div(trans_tlt_embed, self.scale, rounding_mode='trunc').to(device)
        score = torch.einsum('bqnh,bknh->bkqn', trans_src_embed, trans_tlt_embed)  # 8,17,277 ,1
        if (src_mask != None):  # 8,277
            _src_mask = src_mask[:, None, :, None].eq(0)  # 8,1,277,1
            score = score.masked_fill(_src_mask, -float('inf'))  # 8 17 277 1
        # torch.set_printoptions(profile="full")
        # 一行一行做归一化
        attn = self.softmax(score)  # 8,277,17
        attn = torch.where(torch.isnan(attn), torch.full_like(attn, 0), attn).to(device)  # [8, 17, 277, 1]
        attn = torch.squeeze(attn, dim=3)  # 8 17 277]
        attn = F.dropout(attn, p=self.dropout, training=self.training)
        # si 与title的相关性  求和
        attn_similar = torch.einsum('bkq,bkn->bqn', attn, tlt_embed)  # 8,277,512

        attn_output=0.5*src_embed+0.5*attn_similar

        return attn_output

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.w_q.weight)  # 权重初始化



class title_attention_merge(nn.Module):
    def __init__(self, d_model=512, n_head=8, dropout=0.1):
        super(title_attention_merge, self).__init__()
        self.d_model=d_model
        self.similar_attn=similar_attn(d_model=512)
        self.dropout = dropout
        # self.norm = nn.LayerNorm(d_model)
        # self.enc_attn=encode_attn()
        self.line_layer = nn.Linear(d_model, d_model)

        # def reset_parameters(self):
    #     nn.init.xavier_uniform_(self.w_q.weight)  # 权重初始化

    def forward(self,src_embed,tlt_embed,src_mask):
        #求相似分数  [8, 277, 512]  [8, 17, 512]
        # residual = src_embed

        attn_output=self.similar_attn(src_embed,tlt_embed,src_mask)
        # print(src_embed.shape)
        # print(attn_output.shape)
        attn = F.dropout(attn_output, p=self.dropout, training=self.training)
        encoder_output=self.line_layer(attn)
        # encoder_output= self.norm(residual + attn)
        return encoder_output

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.line_layer.weight)  # 权重初始化



    # def reset_parameters(self):
    #     nn.init.xavier_uniform_(self.w_q.weight)  # 权重初始化

class encode_attn(nn.Module):
    def __init__(self,d_model=512,dropout=0.1):
        super(encode_attn, self).__init__()
        self.line_layer=nn.Linear(d_model,d_model)     # 线性层
        self.dropout=dropout
        self.reset_parameters()

    def forward(self,merge):
        # merge=src_embed+attn
        #残差x = self.norm(residual + x)
        l_out=merge+self.line_layer(merge)
        l_out=F.dropout(l_out, p=self.dropout, training=self.training)
        # 残差和层规范化
        mean = l_out.mean(-1, keepdim=True)
        std = l_out.std(-1, keepdim=True)
        output=(l_out - mean) / (std + 1e-9)
        return output

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.line_layer.weight)  # 权重初始化

    
class TransformerSeq2SeqEncoder(nn.Module):
    def __init__(self, embed, pos_embed=None, num_layers=6, d_model=512, n_head=8, dim_ff=2048, dropout=0.1):
        """
        基于Transformer的Encoder
        :param embed: encoder输入token的embedding
        :param nn.Module pos_embed: position embedding
        :param int num_layers: 多少层的encoder
        :param int d_model: 输入输出的维度
        :param int n_head: 多少个head
        :param int dim_ff: FFN中间的维度大小
        :param float dropout: Attention和FFN的dropout大小
        """
        super(TransformerSeq2SeqEncoder, self).__init__()
        self.embed = embed
        self.embed_scale = math.sqrt(d_model)      # d_model=512,  求平方根
        self.pos_embed = pos_embed
        self.num_layers = num_layers
        self.d_model = d_model
        self.n_head = n_head
        self.dim_ff = dim_ff
        self.dropout = dropout

        self.input_fc = nn.Linear(self.embed.embedding_dim, d_model)
        self.layer_stacks = nn.ModuleList([TransformerSeq2SeqEncoderLayer(d_model, n_head, dim_ff, dropout)
                                           for _ in range(num_layers)])


        self.layer_norm = nn.LayerNorm(d_model)

    @classmethod
    def from_opt(cls, opt, embed, pos_embed):  #  有就用用户定义的， 没有就用预定义的
        return cls(embed,
                   pos_embed,
                   num_layers=opt.enc_layers,
                   d_model=opt.d_model,
                   n_head=opt.n_head,
                   dim_ff=opt.dim_ff,
                   dropout=opt.dropout)

    def forward(self, src, src_lens, src_mask, tlt, tlt_lens, tlt_mask):
        """
        :param tokens: batch x max_len
        :param seq_len: [batch]
        :return: bsz x max_len x d_model, bsz x max_len(为0的地方为padding)

        """

        # if(src.device.type=='cpu'):
        #     print("No")
        #     device = torch.device("cuda:0" if torch.cuda.is_available() else exit()) # 单GPU或者CPU
        #     src.to(device)
        #     src_mask.to(device)
        # device = torch.device("cuda:0" if torch.cuda.is_available() else exit())

        # 张量拼接

        input = torch.cat((tlt, src), dim=1)
        # print(tlt.is_cuda)
        # print(tlt.is_cuda)

        # x = self.embed(src) * self.embed_scale    # batch, seq, dim
        x = self.embed(input) * self.embed_scale

        batch_size, max_srctlt_len, _ = x.size()
        _, max_tlt_len = tlt.size()
        _, max_src_len = src.size()
        # print(batch_size, max_src_len, _)     #[8 277 512]

        input_mask = torch.cat((tlt_mask, src_mask), dim=1)
        device = x.device
        if self.pos_embed is not None:
            position = torch.arange(1, max_srctlt_len + 1).unsqueeze(0).long().to(device)
            x += self.pos_embed(position)

        x = self.input_fc(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        for layer in self.layer_stacks:
            # TransformerSeq2SeqEncoderLayer 6次
            # x = layer(x, src_mask)
            x = layer(x, input_mask)

        x = self.layer_norm(x)

        # 拆分

        tlt_embed, src_embed = torch.split(x, [max_tlt_len, max_src_len], dim=1)
        # print( tlt_embed.shape)
        # print(src_embed.shape)
        # exit()

        return tlt_embed, src_embed

