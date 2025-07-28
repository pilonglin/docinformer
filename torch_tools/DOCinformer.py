# Authors: Yonghao Song <eeyhsong@gmail.com>
#
# License: BSD (3-clause)
import warnings
import math
import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
import random
from einops import rearrange
from einops.layers.torch import Rearrange
from torch import nn, Tensor

from braindecode.models.base import EEGModuleMixin

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
xyz_df = pd.read_csv('F:\\PLL\\62.xyz',sep='\t', header=None, names=["ID", "X", "Y", "Z", "ChName"])
default_standard_chs = ['Fpz','P8', 'Oz', 'TP9', 'C4', 'FC6', 'F4', 'CP6', 'CP1', 'P4',  
           'T7', 'FC1', 'O1', 'TP5', 'FC2', 'CP2', 'C3', 'P7', 'FC5', 'F7', 
           'P3', 'Fp2', 'O2', 'F8', 'Pz', 'CP5', 'F3', 'Fp1', 'T8','Fz']

class CoordinateSinPositionEncoding(nn.Module):
    def __init__(self, d_model,standard_chs, base=5000):
        super().__init__()
        self.max_sequence_length = len(standard_chs)  # 最大序列长度
        self.d_model = d_model  # 模型的维度
        self.standard_chs = standard_chs
        # 归一化坐标矩阵的每一行
        self.normalized_coordinates = self.normalize_coordinates()
        self.base = base

    def normalize_coordinates(self):
        # 对每一行坐标进行0-1归一化
        xyz_df = pd.read_csv("62.xyz",sep='\t', header=None, names=["ID", "X", "Y", "Z", "ChName"])
        xyz_df = xyz_df.map(lambda x: x.strip() if isinstance(x, str) else x)
        # 筛选通道位置
        standard_xyz_df  = xyz_df[xyz_df["ChName"].isin(self.standard_chs)]
        # 使用merge按照standard_chs列表的顺序进行排序
        standard_chs_df = pd.DataFrame({'ChName': self.standard_chs})
        standard_xyz_df = standard_chs_df.merge(standard_xyz_df, on='ChName', how='left')
        # 位置归一化
        all_coordinate = xyz_df[['X', 'Y', 'Z']].to_numpy().transpose()
        coordinate = standard_xyz_df[['X', 'Y', 'Z']].to_numpy().transpose()
        normalized_array = coordinate - np.min(coordinate, axis=1, keepdims=True)
        normalized_array /= np.max(all_coordinate, axis=1, keepdims=True) - np.min(all_coordinate, axis=1, keepdims=True)
        return torch.tensor(normalized_array)

    def forward(self):
        # 初始化位置编码矩阵，大小为(max_sequence_length, d_model)
        pe = torch.zeros(self.max_sequence_length, self.d_model, dtype=torch.float)
        
        # 计算衰减因子 alpha
        div_term = torch.exp(torch.arange(0, self.d_model, dtype=torch.float) * -(math.log(self.base) / self.d_model))
        
        # 使用坐标的正弦值填充位置编码矩阵
        for i in range(0, self.d_model, 3):  # 每次循环处理3个维度
            for j in range(3):  # 对于每个坐标
                if i + j < self.d_model:  # 确保不超过 d_model 的维度
                    sin_values = torch.sin(self.normalized_coordinates[j, :] * div_term[i + j])
                    pe[:, i + j] = sin_values
        return pe


class SinPositionEncoding(nn.Module):
    def __init__(self, max_sequence_length, d_model, base=5000):
        super().__init__()
        self.max_sequence_length = max_sequence_length
        self.d_model = d_model
        self.base = base

    def forward(self):
        pe = torch.zeros(self.max_sequence_length, self.d_model, dtype=torch.float)  # size(max_sequence_length, d_model)
        exp_1 = torch.arange(self.d_model // 2, dtype=torch.float)  # 初始化一半维度，sin位置编码的维度被分为了两部分
        exp_value = exp_1 / (self.d_model / 2)

        alpha = 1 / (self.base ** exp_value)  # size(dmodel/2)
        out = torch.arange(self.max_sequence_length, dtype=torch.float)[:, None] @ alpha[None, :]  # size(max_sequence_length, d_model/2)
        embedding_sin = torch.sin(out)
        embedding_cos = torch.cos(out)

        pe[:, 0::2] = embedding_sin  # 奇数位置设置为sin
        pe[:, 1::2] = embedding_cos  # 偶数位置设置为cos
        return pe

class DOCinformer(EEGModuleMixin, nn.Module):
    """
    Parameters
    ----------
    n_filters_time: int
        Number of temporal filters, defines also embedding size.
    filter_time_length: int
        Length of the temporal filter.
    pool_time_length: int
        Length of temporal pooling filter.
    pool_time_stride: int
        Length of stride between temporal pooling filters.
    drop_prob: float
        Dropout rate of the convolutional layer.
    att_depth: int
        Number of self-attention layers.
    att_heads: int
        Number of attention heads.
    att_drop_prob: float
        Dropout rate of the self-attention layer.
    final_fc_length: int | str
        The dimension of the fully connected layer.
    return_features: bool
        If True, the forward method returns the features before the
        last classification layer. Defaults to False.
    activation: nn.Module
        Activation function as parameter. Default is nn.ELU
    activation_transfor: nn.Module
        Activation function as parameter, applied at the FeedForwardBlock module
        inside the transformer. Default is nn.GeLU

    References
    ----------
    .. [song2022] Song, Y., Zheng, Q., Liu, B. and Gao, X., 2022. EEG
       conformer: Convolutional transformer for EEG decoding and visualization.
       IEEE Transactions on Neural Systems and Rehabilitation Engineering,
       31, pp.75-719. https://ieeexplore.ieee.org/document/9991178
    .. [ConformerCode] Song, Y., Zheng, Q., Liu, B. and Gao, X., 2022. EEG
       conformer: Convolutional transformer for EEG decoding and visualization.
       https://github.com/eeyhsong/EEG-Conformer.
    """

    def __init__(
        self,
        n_outputs=None,
        n_filters_time=40,
        filter_time_length=50,
        pool_time_length=75,
        pool_time_stride=15,
        drop_prob=0.5,
        att_depth=2,
        att_heads=2,
        t_att_depth=1,
        t_att_heads=1,
        att_drop_prob=0.5,
        return_features=False,
        activation: nn.Module = nn.ELU,
        activation_transfor: nn.Module = nn.GELU,
        n_times=None,
        chs_info=None,
        input_window_seconds=None,
        sfreq=None,
        standard_chs = default_standard_chs,
        channel_drop_prob = 0.5,
        positionEncoding = None
    ):
        super().__init__(
            n_outputs=n_outputs,
            n_chans=len(standard_chs),
            chs_info=chs_info,
            n_times=n_times,
            input_window_seconds=input_window_seconds,
            sfreq=sfreq,
        )
        self.standard_chs = standard_chs
        self.select_chs = standard_chs
        self.channel_drop_prob = channel_drop_prob
        self.emb_size = n_filters_time
        self.mapping = {
            "classification_head.fc.6.weight": "final_layer.final_layer.0.weight",
            "classification_head.fc.6.bias": "final_layer.final_layer.0.bias",
        }

        del n_outputs, chs_info, n_times, input_window_seconds, sfreq

        self.patch_embedding = _PatchEmbedding(
            n_filters_time=n_filters_time,
            filter_time_length=filter_time_length,
            pool_time_length=pool_time_length,
            stride_avg_pool=pool_time_stride,
            drop_prob=drop_prob,
            activation=activation,
        )
        self.spatial = _SperateTransform(
            emb_size = self.emb_size,
            positionEncoding = positionEncoding,
            standard_chs = self.standard_chs,
            att_depth=att_depth,
            att_heads=att_heads,
        )
        
        self.s2t = nn.Sequential(
            Rearrange("b ch emb t -> b t (emb ch)"),
        )
        self.s2t.to(device)

        self.temporal = nn.Sequential(
            nn.TransformerEncoder( 
                nn.TransformerEncoderLayer(batch_first = True,d_model= self.emb_size, nhead=t_att_heads, dim_feedforward=n_filters_time*4, dropout=att_drop_prob),
                num_layers=t_att_depth
            ),
            # nn.AdaptiveMaxPool1d(1),
        )
        self.temporal.to(device)

        self.fc = _FullyConnected(
            final_fc_length=self.get_fc_size(), activation=activation
        )

        self.final_layer = _FinalLayer(
            n_classes=self.n_outputs,
            return_features=return_features,
        )

    def forward(self, x: Tensor) -> Tensor:
        x = torch.unsqueeze(x, dim=1)  # add one extra dimension
        x = self.patch_embedding(x)
        if self.training and self.channel_drop_prob > 0.0:
            x = self.spatial(x,self.channel_drop_prob)
        else:
            x = self.spatial(x)
        x = self.s2t(x)
        x = self.temporal(x)
        x = self.fc(x)
        x = self.final_layer(x)
        return x

    def get_fc_size(self):
        x = self.patch_embedding(torch.ones((1, 1, len(self.standard_chs), self.n_times)))
        x = self.spatial(x)
        x = self.s2t(x)
        out = self.temporal(x).cpu().data.numpy()
        return out.shape[1]*out.shape[2]
    
    def select_channel(self,select_chs):
        # 将列表转换为集合
        select_set = set(select_chs)
        standard_set = set(self.standard_chs)
        # 使用 issubset 方法判断是否为子集
        is_subset = select_set.issubset(standard_set)
        if not is_subset:
            warnings.warn("包含不在训练集里的通道")
        self.spatial.change_channel(select_chs)
        return 
    
    
class _SperateTransform(nn.Module):
    def __init__(
        self,
        positionEncoding,
        emb_size,
        standard_chs,
        att_depth = 2,
        att_heads = 2,
        att_drop_prob = 0.5,
    ):
        super().__init__()
        self.emb_size = emb_size
        self.positionEncoding = positionEncoding
        self.standard_chs = standard_chs
        if self.positionEncoding == 'sincos':
            self.positional_ch_encoding = SinPositionEncoding(d_model=emb_size, max_sequence_length=len(self.standard_chs)).forward().to(device)
        elif self.positionEncoding == 'coordinate':
            self.positional_ch_encoding = CoordinateSinPositionEncoding(d_model=emb_size, standard_chs = self.standard_chs).forward().to(device)
        else :
            self.positional_ch_encoding = None
        # 创建一个包含多个编码器层的 Transformer 编码器
        encoder_layer = nn.TransformerEncoderLayer(batch_first = True,d_model=emb_size, nhead=att_heads, dim_feedforward=emb_size*4, dropout=att_drop_prob)
        self.transformerEncoder = nn.TransformerEncoder(encoder_layer, num_layers=att_depth)
        self.transformerEncoder.to(device)
        # 池化
        self.pool = nn.Sequential(
            Rearrange("b ch emb -> b emb ch"),
            nn.AdaptiveAvgPool1d(1),
            Rearrange("b emb ch -> b ch emb"),
        )
        self.pool.to(device)

    def forward(self, x: Tensor , ch_dropout = 0.0) -> Tensor:
        # 将x的最后一个维度w分割成w个独立的序列，每个时间点的通道序列独立处理
        spatial_sequences = torch.unbind(x, dim=-1)  # 列表，每个元素形状为 [batch_size, e, h]
        # 对每个序列独立使用TransformerEncoderBlock处理
        transformed_sequences = []
        for spatial_seq in spatial_sequences:
            seq = spatial_seq.to(device) 
            if not self.positional_ch_encoding == None : 
                seq = seq + self.positional_ch_encoding
            if ch_dropout > 0.0:
                mask = torch.empty(x.shape[1], dtype=torch.float32).uniform_() < ch_dropout     # 生成通道掩码
                seq = seq[:, ~mask, :]
            seq = self.transformerEncoder(seq)
            seq = self.pool(seq)
            transformed_sequences.append(seq)
        
        # 将处理后的序列重新组合
        x = torch.stack(transformed_sequences, dim=-1).to(device)
        return x
    
    def change_channel(self,select_chs):
        if self.positionEncoding == 'sincos':
            ch_index_map = {ch: index for index, ch in enumerate(self.standard_chs)}
            self.positional_ch_encoding = torch.stack([self.positional_ch_encoding[ch_index_map[ch]] for ch in select_chs])
        elif  self.positionEncoding == 'coordinate':
            self.positional_ch_encoding = CoordinateSinPositionEncoding(d_model=self.emb_size, standard_chs = select_chs).forward().to(device)
        return


class _PatchEmbedding(nn.Module):
    """Patch Embedding.

    The authors used a convolution module to capture local features,
    instead of position embedding.

    Parameters
    ----------
    n_filters_time: int
        Number of temporal filters, defines also embedding size.
    filter_time_length: int
        Length of the temporal filter.
    n_channels: int
        Number of channels to be used as number of spatial filters.
    pool_time_length: int
        Length of temporal poling filter.
    stride_avg_pool: int
        Length of stride between temporal pooling filters.
    drop_prob: float
        Dropout rate of the convolutional layer.

    Returns
    -------
    x: torch.Tensor
        The output tensor of the patch embedding layer.
    """

    def __init__(
        self,
        n_filters_time,
        filter_time_length,
        pool_time_length,
        stride_avg_pool,
        drop_prob,
        activation: nn.Module = nn.ELU,
    ):
        super().__init__()

        
        # 计算膨胀卷积的填充，确保输出尺寸与标准卷积相同
        dilation_padding_2 = filter_time_length - 1
        dilation_padding_3 = (3 * (filter_time_length - 1)) // 2

        # self.shallownet_d3 = nn.Sequential(
        #     nn.Conv2d(1, n_filters_time, (1, filter_time_length), (1, 1), dilation=(1, 3), padding=(0, dilation_padding_3)),
        #     nn.BatchNorm2d(num_features=n_filters_time),
        #     activation(),
        # )

        self.shallownet_d2 = nn.Sequential(
            nn.Conv2d(1, n_filters_time, (1, filter_time_length), (1, 1), dilation=(1, 2), padding=(0, dilation_padding_2)),
            nn.BatchNorm2d(num_features=n_filters_time),
            activation(),
        )

        self.shallownet = nn.Sequential(
            nn.Conv2d(1, n_filters_time, (1, filter_time_length), (1, 1), padding=(0, (filter_time_length - 1) // 2)),
            nn.BatchNorm2d(num_features=n_filters_time),
            activation(),
        )

        self.projection = nn.Sequential(
            # 时间池化
            nn.MaxPool2d(
                kernel_size=(1, pool_time_length), stride=(1, stride_avg_pool)
            ),
            # Dropout
            nn.Dropout(drop_prob),
            # 压缩通道
            nn.Conv2d(
                n_filters_time*2, n_filters_time, (1, 1), stride=(1, 1)
            ),
            # 维度交换
            Rearrange("b emb ch t -> b ch emb t"),
        )

    def forward(self, x: Tensor) -> Tensor:
        x = torch.cat((self.shallownet(x),self.shallownet_d2(x)), dim=1)
        x = self.projection(x)
        return x


class _FullyConnected(nn.Module):
    def __init__(
        self,
        final_fc_length,
        drop_prob_1=0.5,
        drop_prob_2=0.3,
        out_channels=256,
        hidden_channels=32,
        activation: nn.Module = nn.ELU,
    ):
        """Fully-connected layer for the transformer encoder.

        Parameters
        ----------
        final_fc_length : int
            Length of the final fully connected layer.
        n_classes : int
            Number of classes for classification.
        drop_prob_1 : float
            Dropout probability for the first dropout layer.
        drop_prob_2 : float
            Dropout probability for the second dropout layer.
        out_channels : int
            Number of output channels for the first linear layer.
        hidden_channels : int
            Number of output channels for the second linear layer.
        return_features : bool
            Whether to return input features.
        """

        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(final_fc_length, out_channels),
            activation(),
            nn.Dropout(drop_prob_1),
            nn.Linear(out_channels, hidden_channels),
            activation(),
            nn.Dropout(drop_prob_2),
        )

    def forward(self, x):
        x = x.contiguous().view(x.size(0), -1)
        out = self.fc(x)
        return out


class _FinalLayer(nn.Module):
    def __init__(
        self,
        n_classes,
        hidden_channels=32,
        return_features=False,
    ):
        """Classification head for the transformer encoder.

        Parameters
        ----------
        n_classes : int
            Number of classes for classification.
        hidden_channels : int
            Number of output channels for the second linear layer.
        return_features : bool
            Whether to return input features.
        """

        super().__init__()
        self.final_layer = nn.Sequential(
            nn.Linear(hidden_channels, n_classes),
        )
        self.return_features = return_features
        classification = nn.Identity()
        if not self.return_features:
            self.final_layer.add_module("classification", classification)

    def forward(self, x):
        if self.return_features:
            out = self.final_layer(x)
            return out, x
        else:
            out = self.final_layer(x)
            return out