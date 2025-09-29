'''
This version panoformer keep almost structure the same with original panoformer.

The parameter on this version is less than the original panoformer.

The only structure difference is that : current version panoformer's downsample layer have a relu activation functions.
'''
import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
from timm.models.layers import DropPath,  trunc_normal_
import torch.nn.functional as F
from einops import rearrange
from PSA import *

from equisamplingpoint import genSamplingPattern



########### feed-forward network #############
class LeFF(nn.Module):
    def __init__(self, dim=32, hidden_dim=128, act_layer=nn.GELU):
        super().__init__()
        self.linear1 = nn.Sequential(nn.Linear(dim, hidden_dim),
                                     act_layer())
        self.dwconv = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, groups=hidden_dim, kernel_size=3, stride=1, padding=0),
            act_layer())
        self.linear2 = nn.Sequential(nn.Linear(hidden_dim, dim))

    def forward(self, x, H, W):
        # bs x hw x c
        hh = H

        x = self.linear1(x)
        # spatial restore
        x = rearrange(x, ' b (h w) (c) -> b c h w ', h=hh, w=hh * 2)

     
        x = F.pad(x, (1, 1, 0, 0), mode='circular')  # width
        x = F.pad(x, (0, 0, 1, 1))

        x = self.dwconv(x)
        # flaten
        x = rearrange(x, ' b c h w -> b (h w) c', h=hh, w=hh * 2)

        x = self.linear2(x)

        return x
    
# Downsample Block
class Downsample(nn.Module):
    def __init__(self, in_channel, out_channel, input_resolution=None):
        super(Downsample, self).__init__()
        self.input_resolution = input_resolution
        self.conv = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            #nn.BatchNorm2d(out_channel)

        )

    def forward(self, x):
        B, L, C = x.shape
        # import pdb;pdb.set_trace()
        H, W = self.input_resolution
        x = x.transpose(1, 2).contiguous().view(B, C, H, W)
        x = F.pad(x, (1, 1, 0, 0), mode='circular')  # width
        x = F.pad(x, (0, 0, 1, 1))
        out = self.conv(x).flatten(2).transpose(1, 2).contiguous()  # B H*W C
        return out
    
# Upsample Block
class Upsample(nn.Module):
    def __init__(self, in_channel, out_channel, input_resolution=None):
        super(Upsample, self).__init__()
        self.input_resolution = input_resolution
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(in_channel, out_channel, kernel_size=2, stride=2),
        )

    def forward(self, x):
        B, L, C = x.shape
        H, W = self.input_resolution
        x = x.transpose(1, 2).contiguous().view(B, C, H, W)
        out = self.deconv(x).flatten(2).transpose(1, 2).contiguous()  # B H*W C
        return out
    
# Input Projection
class InputProj(nn.Module):
    def __init__(self, in_channel=3, out_channel=64, kernel_size=3, stride=1, norm_layer=None, act_layer=nn.LeakyReLU):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=kernel_size, stride=stride, padding=0),
            act_layer()
        )
        if norm_layer is not None:
            self.norm = norm_layer(out_channel)
        else:
            self.norm = None

    def forward(self, x):
        B, C, H, W = x.shape
        x = F.pad(x, (3 // 2, 3 // 2, 0, 0), mode='circular')  # width
        x = F.pad(x, (0, 0, 3 // 2, 3 // 2))
        x = self.proj(x).flatten(2).transpose(1, 2).contiguous()  # B H*W C
        if self.norm is not None:
            x = self.norm(x)
        return x

class OutputProj(nn.Module):
    def __init__(self, in_channel=64, out_channel=1, kernel_size=3, stride=1, norm_layer=None, act_layer=None,
                 input_resolution=None):
        super().__init__()
        self.input_resolution = input_resolution
        self.proj = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=kernel_size, stride=stride, padding=1),
        )
        if act_layer is not None:
            self.proj.add_module(act_layer(inplace=True))
        if norm_layer is not None:
            self.norm = norm_layer(out_channel)
        else:
            self.norm = None

    def forward(self, x):
        B, L, C = x.shape
        H, W = self.input_resolution
        x = x.transpose(1, 2).view(B, C, H, W)
        x = F.interpolate(x, scale_factor=2, mode='nearest')#for 1024*512
        #x = F.pad(x, (3 // 2, 3 // 2, 0, 0), mode='circular')  # width
        #x = F.pad(x, (0, 0, 3 // 2, 3 // 2))
        x = self.proj(x)
        if self.norm is not None:
            x = self.norm(x)
        return x

#########################################
########### LeWinTransformer #############
class PanoformerBlock(nn.Module):
    def __init__(self, dim, input_resolution, num_heads, win_size=8, shift_size=4,
                 mlp_ratio=4.,drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm, 
                ref_point = None):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.win_size = win_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        self.ref_point = ref_point #generate_ref_points(self.input_resolution[1], self.input_resolution[0])
        if min(self.input_resolution) <= self.win_size:
            self.shift_size = 0
            self.win_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.win_size, "shift_size must in 0-win_size"

        self.norm1 = norm_layer(dim)

        self.dattn = PanoSelfAttention(num_heads, dim, k=9, last_feat_height=self.input_resolution[0], last_feat_width=self.input_resolution[1], scales=1, dropout=0, need_attn=False)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = LeFF(dim, mlp_hidden_dim, act_layer=act_layer)

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, num_heads={self.num_heads}, " \
               f"win_size={self.win_size}, shift_size={self.shift_size}, mlp_ratio={self.mlp_ratio}"

    def forward(self, x):
        B, L, C = x.shape
        H, W = self.input_resolution
        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)
        # W-MSA/SW-MSA
        x = self.dattn(x, x.unsqueeze(0), self.ref_point.repeat(B, 1, 1, 1, 1))  # nW*B, win_size*win_size, C

        x = x.view(B, H * W, C)

        # FFN
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x), H, W))
        return x

########### Basic layer of Uformer ################
class BasicPanoformerLayer(nn.Module):
    def __init__(self, dim, output_dim, input_resolution, depth, num_heads, win_size,
                 mlp_ratio=4., 
                 drop_path=0.1, norm_layer=nn.LayerNorm, use_checkpoint=False,
                 ref_point = None):

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint
        # build blocks
        self.blocks = nn.ModuleList([
            PanoformerBlock(dim=dim, input_resolution=input_resolution,
                                  num_heads=num_heads, win_size=win_size,
                                  shift_size=0 if (i % 2 == 0) else win_size // 2,
                                  mlp_ratio=mlp_ratio,
                                  drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                  norm_layer=norm_layer, 
                                ref_point=ref_point)
            for i in range(depth)])

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, depth={self.depth}"

    def forward(self, x):
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)
        return x
    
