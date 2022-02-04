import math

from numpy.lib.arraypad import pad
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import models

dtype = torch.cuda.FloatTensor
dtype_long = torch.cuda.LongTensor

def positionalencoding2d(feat_size, height, width):
    if feat_size % 4 != 0:
        raise ValueError("Cannot use sin/cos positional encoding with "
                            "odd dimension (got dim={:d})".format(feat_size))
    pe = torch.zeros(feat_size, height, width)
    # Each dimension use half of feat_size
    feat_size = int(feat_size / 2)
    div_term = torch.exp(torch.arange(0., feat_size, 2) *
                            -(math.log(10000.0) / feat_size))
    pos_w = torch.arange(0., width).unsqueeze(1)
    pos_h = torch.arange(0., height).unsqueeze(1)
    pe[0:feat_size:2, :, :] = torch.sin(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
    pe[1:feat_size:2, :, :] = torch.cos(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
    pe[feat_size::2, :, :] = torch.sin(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)
    pe[feat_size + 1::2, :, :] = torch.cos(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)
    return pe

class PositionalEncoding1D(nn.Module):
    def __init__(self, channels):
        """
        :param channels: The last dimension of the tensor you want to apply pos emb to.
        """
        super(PositionalEncoding1D, self).__init__()
        self.channels = channels
        inv_freq = 1. / (10000 ** (torch.arange(0, channels, 2).float() / channels))
        self.register_buffer('inv_freq', inv_freq)

    def forward(self, tensor):
        """
        :param tensor: A 3d tensor of size (batch_size, x, ch)
        :return: Positional Encoding Matrix of size (batch_size, x, ch)
        """
        if len(tensor.shape) != 3:
            raise RuntimeError("The input tensor has to be 3d!")
        batch_size, x, orig_ch = tensor.shape
        pos_x = torch.arange(x, device=tensor.device).type(self.inv_freq.type())
        sin_inp_x = torch.einsum("i,j->ij", pos_x, self.inv_freq)
        emb_x = torch.cat((sin_inp_x.sin(), sin_inp_x.cos()), dim=-1)
        emb = torch.zeros((x,self.channels),device=tensor.device).type(tensor.type())
        emb[:,:self.channels] = emb_x

        return emb[None,:,:orig_ch].repeat(batch_size, 1, 1)

class ResidualBlock(nn.Module):
  def __init__(self, in_channels, out_channels, stride, padding):
    super(ResidualBlock, self).__init__()
    self.in_channels = in_channels
    self.out_channels = out_channels

    self.downsampling = nn.Sequential(
                          nn.Conv2d(self.in_channels, self.out_channels, kernel_size=3, stride=stride, padding=padding),
                          nn.BatchNorm2d(self.out_channels)
                        )

    self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
    self.bn1 = nn.BatchNorm2d(out_channels)
    self.relu = nn.ReLU(inplace=True)
    self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=padding)
    self.bn2 = nn.BatchNorm2d(out_channels)

  def forward(self, x):
    residual = x
    out = self.conv1(x)
    out = self.bn1(out)
    out = self.relu(out)
    out = self.conv2(out)
    out = self.bn2(out)
    if self.in_channels != self.out_channels:
      residual = self.downsampling(residual)
    out += residual
    out = self.relu(out)
    return out


class ResidualNet(nn.Module):
  def __init__(self, block_sizes):
    super(ResidualNet, self).__init__()
    self.res_blocks = [ResidualBlock(in_c, out_c, stride, padding).cuda() for (in_c, out_c, stride, padding) in block_sizes[:-1]]
    in_chanels = block_sizes[-1][0]
    out_chanels = block_sizes[-1][1]
    self.conv1 = nn.Conv2d(in_chanels, out_chanels, kernel_size=3, stride=1, padding=1)
    self.bn1 = nn.BatchNorm2d(out_chanels)
    self.relu = nn.ReLU(inplace=True)
    self.conv2 = nn.Conv2d(out_chanels, out_chanels, kernel_size=3, stride=1, padding=1)
    self.bn2 = nn.BatchNorm2d(out_chanels)

  def forward(self, x):
    out = x
    for res_block in self.res_blocks:
      out = res_block(out)
    out = self.conv1(out)
    out = self.bn1(out)
    out = self.relu(out)
    out = self.conv2(out)
    out = self.bn2(out)
    return out

class MultiHeadAttention(nn.Module):
    def __init__(self, n_heads, feat_size):
        super(MultiHeadAttention, self).__init__()

        assert feat_size % n_heads == 0
        self.feat_size = feat_size
        self.n_heads = n_heads
        self.head_size = feat_size // n_heads

        self.q_fc = nn.Linear(feat_size, feat_size)
        self.k_fc = nn.Linear(feat_size, feat_size)
        self.v_fc = nn.Linear(feat_size, feat_size)

        self.final_fc = nn.Linear(feat_size, feat_size)

    def forward(self, Q, K, V):
        """
            Q : query tensor [b, c, n]
            K : key tensor [b, c, n]
            V : value tensor [b, c, n]
        """
        assert self.feat_size == Q.shape[1] == K.shape[1] == V.shape[1]
        assert Q.shape[0] == K.shape[0] == V.shape[0]
        
        batch_size, c, n = Q.shape

        # [b, c, n] -> [b, n, c]
        Q = Q.permute(0, 2, 1)
        K = K.permute(0, 2, 1)
        V = V.permute(0, 2, 1)

        Q = self.q_fc(Q)
        K = self.k_fc(K)
        V = self.v_fc(V)

        # Split into n heads : [b, n, c] -> [b, n_heads, n, head_size]
        Q = Q.view(batch_size, -1, self.n_heads, self.head_size).permute(0, 2, 1, 3)
        K = K.view(batch_size, -1, self.n_heads, self.head_size).permute(0, 2, 1, 3)
        V = V.view(batch_size, -1, self.n_heads, self.head_size).permute(0, 2, 1, 3)

        score = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_size) #maybe set manual norm factor
        softmax = F.softmax(score, dim=-1)
        attention = torch.matmul(softmax, V)
        
        attention = attention.permute(0, 2, 1, 3).contiguous().view(batch_size, n, self.feat_size)

        attention = self.final_fc(attention)

        attention = attention.permute(0, 2, 1)

        return attention

class FeedForward(nn.Module):
    def __init__(self, feat_size, hidden_size=2048):
        super(FeedForward, self).__init__()
        self.fc1 = nn.Linear(feat_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, feat_size)

    def forward(self, x):
        b, c, n = x.shape
        # [b, c, n] -> [b, n, c]
        x = x.permute(0, 2, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        # [b, n, c] -> [b, c, n]
        x = x.permute(0, 2, 1)
        return x

class Normalization(nn.Module):
    def __init__(self, feat_size, eps=1e-6):
        super(Normalization, self).__init__()
        self.gamma = nn.Parameter(torch.ones(feat_size))
        self.beta = nn.Parameter(torch.zeros(feat_size))
        self.eps = eps

    def forward(self, x):
        b, c, n = x.shape
        # [b, c, n] -> [b, n, c]
        x = x.permute(0, 2, 1)
        norm = self.gamma * ((x - x.mean(dim=-1, keepdim=True)) / torch.sqrt(x.var(dim=-1, keepdim=True) + self.eps)) + self.beta
        # [b, n, c] -> [b, c, n]
        norm = x.permute(0, 2, 1)
        return norm

class TransformerLayer(nn.Module):
    def __init__(self, fn, feat_size, hidden_size=128):
        super(TransformerLayer, self).__init__()
        self.norm1 = Normalization(feat_size)
        self.norm2 = Normalization(feat_size)
        self.attention = fn
        self.fc = FeedForward(feat_size, hidden_size)

    def forward(self, Q, K, V):
        Q2 = self.attention(Q, K, V)
        Q2 = self.norm1(Q + Q2)
        Q3 = self.fc(Q2)
        Q3 = self.norm2(Q2 + Q3)
        return Q3

class ArielNetwork(nn.Module):

  def __init__(self, feat_size, n_heads):
    super(ArielNetwork, self).__init__()

    hidden_dim = 128

    self.pos_encoding = positionalencoding2d(128, 55, 75).unsqueeze(0).cuda()
    self.conv_backbone = ResidualNet([[1, 32, (1, 1), (1, 1)], [32, 32, (1, 1), (1, 1)], [32, 64, (1, 2), (1, 1)], [64, 64, (1, 1), (1, 1)], [64, 128, (1, 2), (1, 1)], [128, 128, (1, 1), (1, 1)]]).cuda()

    self.selfAtt_1 = TransformerLayer(MultiHeadAttention(n_heads, feat_size), feat_size, hidden_dim)
    self.selfAtt_2 = TransformerLayer(MultiHeadAttention(n_heads, feat_size), feat_size, hidden_dim)
    self.selfAtt_3 = TransformerLayer(MultiHeadAttention(n_heads, feat_size), feat_size, hidden_dim)


    self.proj_coord = nn.Sequential(
      nn.Conv1d(2, 64, kernel_size=1, stride=1),
      nn.BatchNorm1d(64),
      nn.ReLU(inplace=True),
      nn.Conv1d(64, 128, kernel_size=1, stride=1),
      nn.BatchNorm1d(128)
    )

    self.crossAtt_coord = TransformerLayer(MultiHeadAttention(n_heads, feat_size), feat_size, hidden_dim)
    self.crossAtt_query = TransformerLayer(MultiHeadAttention(n_heads, feat_size), feat_size, hidden_dim)

    self.output = nn.Sequential(
      nn.Conv2d(feat_size, 1, kernel_size=1, stride=1),
      nn.BatchNorm2d(1),
      nn.ReLU(inplace=True),
      nn.Flatten(),
      nn.Linear(4125, 1024),
      nn.ReLU(inplace=True),
      nn.Linear(1024, 55)
    )

  def forward(self, x):
    pe = self.pos_encoding
    x = self.conv_backbone(x) + pe 

    x = self.selfAtt_1(x, x, x)
    x = self.selfAtt_2(x, x, x)
    x = self.selfAtt_3(x, x, x)

    x = self.output(x)

    return x

class ArielCNN(nn.Module):
  def __init__(self):
    super(ArielCNN, self).__init__()
    
    self.resnet50 = models.resnet50(pretrained=True).cuda()
    
    self.param_branch = nn.Sequential(
      nn.Linear(6, 32),
      # nn.BatchNorm1d(32),
      nn.ReLU(),
      nn.Linear(32, 24),
      # nn.BatchNorm1d(24)
    )

    self.final_linear = nn.Sequential(
      nn.Linear(1024, 512),
      # nn.BatchNorm1d(512),
      nn.ReLU(),
      nn.Linear(512, 55),
      # nn.BatchNorm1d(55)
    )

    self.final_linear_aux = nn.Sequential(
      nn.Linear(1024, 256),
      # nn.BatchNorm1d(256),
      nn.ReLU(),
      nn.Linear(256, 64),
      # nn.BatchNorm1d(64),
      nn.ReLU(),
      nn.Linear(64, 2),
      # nn.BatchNorm1d(2)
    )

  def forward(self, X, X_param):

    X = X.unsqueeze(1).repeat(1, 3, 1, 1)
    
    y_resnet = self.resnet50(X)
    y_param = self.param_branch(X_param)
  
    features = torch.cat((y_resnet, y_param), 1)

    out = self.final_linear(features)
    out_aux = self.final_linear_aux(features)

    return out, out_aux

class CrossAttentionNet(nn.Module):
  
  def __init__(self, nb_selfAtt=3):
    super(CrossAttentionNet, self).__init__()
    self.out_query = torch.nn.Parameter(torch.randn(1, 55, 1))
    
    self.conv_in = nn.Sequential(
      nn.Conv1d(55, 64, 3, padding=1),
      nn.BatchNorm1d(64),
      nn.ReLU(),
      nn.Conv1d(64, 128, 1),
      nn.BatchNorm1d(128),
      nn.ReLU()
    )
    
    self.selfAtt_layers = nn.ModuleList([
      TransformerLayer(MultiHeadAttention(4, 128), 128) for _ in range(nb_selfAtt)
    ])
    
    self.conv_out = nn.Sequential(
      nn.Conv1d(128, 64, 1, padding=1),
      nn.BatchNorm1d(64),
      nn.ReLU(),
      nn.Conv1d(64, 55, 1),
      nn.BatchNorm1d(55)
    )
    
  def forward(self, x):
    
    b, c, n = x.shape
    query_out = self.out_query.repeat(b, 1, 1)
    x = torch.cat((x, query_out), 2)
    
    out = self.conv_in(x)
    for attLayer in self.selfAtt_layers:
      out = attLayer(out, out, out)
    
    out = self.conv_out(out)
    
    return out[:,:,-1].squeeze(-1)
  

class CrossAttentionNet_extended(nn.Module):
  
  def __init__(self, nb_selfAtt=3):
    super(CrossAttentionNet_extended, self).__init__()
    
    self.out_query = torch.nn.Parameter(torch.randn(1, 55, 1))
    self.pos_encoding = PositionalEncoding1D(128)
    
    
    self.conv_in = nn.Sequential(
      nn.Conv1d(55, 64, 3, padding=1),
      nn.BatchNorm1d(64),
      nn.ReLU(),
      nn.Conv1d(64, 128, 1),
      nn.BatchNorm1d(128),
      nn.ReLU()
    )
    
    self.selfAtt_layers = nn.ModuleList([
      TransformerLayer(MultiHeadAttention(4, 128), 128) for _ in range(nb_selfAtt)
    ])
    
    self.conv_out = nn.Sequential(
      nn.Conv1d(128, 64, 1, padding=1),
      nn.BatchNorm1d(64),
      nn.ReLU(),
      nn.Conv1d(64, 55, 1),
      nn.BatchNorm1d(55)
    )
    
    self.final_fc = nn.Sequential(
      nn.Linear(55+6, 64),
      nn.ReLU(),
      nn.Linear(64, 64),
      nn.ReLU(),
      nn.Linear(64, 55)
    )
    
  def forward(self, x, x2):
    
    b, c, n = x.shape
    query_out = self.out_query.repeat(b, 1, 1)
    x = torch.cat((x, query_out), 2)
    
    out = self.conv_in(x)
    
    pe = self.pos_encoding(out.permute(0, 2, 1)).permute(0, 2, 1)
    out = out + pe
    
    for attLayer in self.selfAtt_layers:
      out = attLayer(out, out, out)
    
    out = self.conv_out(out)
    
    out = torch.cat([out[:,:,-1].squeeze(-1), x2], dim=1)
    out = self.final_fc(out)
    return out

class CrossAttentionNet_extended2(nn.Module):
  
  def __init__(self, nb_selfAtt=3):
    super(CrossAttentionNet_extended2, self).__init__()
    self.out_query = torch.nn.Parameter(torch.randn(1, 55, 1))
    
    self.conv_in = nn.Sequential(
      nn.Conv1d(55, 64, 3, padding=1),
      nn.BatchNorm1d(64),
      nn.ReLU(),
      nn.Conv1d(64, 128, 1),
      nn.BatchNorm1d(128),
      nn.ReLU()
    )
    
    self.crossAtt_layer = TransformerLayer(MultiHeadAttention(1, 1), 1)
    
    self.selfAtt_layers = nn.ModuleList([
      TransformerLayer(MultiHeadAttention(4, 128), 128) for _ in range(nb_selfAtt)
    ])
    
    self.conv_out = nn.Sequential(
      nn.Conv1d(128, 64, 1, padding=1),
      nn.BatchNorm1d(64),
      nn.ReLU(),
      nn.Conv1d(64, 55, 1),
      nn.BatchNorm1d(55)
    )
    
    self.final_fc = nn.Sequential(
      nn.Linear(55, 64),
      nn.ReLU(),
      nn.Linear(64, 64),
      nn.ReLU(),
      nn.Linear(64, 55)
    )
    
  def forward(self, x, x2):
    
    b, c, n = x.shape
    query_out = self.out_query.repeat(b, 1, 1)
  
    query_out = self.crossAtt_layer(query_out.permute(0, 2, 1), x2.unsqueeze(1), x2.unsqueeze(1)).permute(0, 2, 1)
    
    x = torch.cat((x, query_out), 2)
    
    out = self.conv_in(x)
    for attLayer in self.selfAtt_layers:
      out = attLayer(out, out, out)
    
    out = self.conv_out(out)
    
    out = out[:,:,-1].squeeze(-1)
    out = self.final_fc(out)
    return out


if __name__ == '__main__':
  model = CrossAttentionNet_extended2().cuda()
  
  x = torch.randn(4, 55, 300).cuda()
  x2 = torch.randn(4, 6).cuda()
  y = model(x, x2)
  print(x.shape, x2.shape, y.shape)