import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from timm.models.layers import DropPath,  trunc_normal_
from equisamplingpoint import genSamplingPattern


from model_unit import Downsample, Upsample, InputProj, OutputProj,  BasicPanoformerLayer



class Segmenter(nn.Module):
    def __init__(self, in_channel=64, num_class=2, kernel_size=3, stride=1,
                 input_resolution=None):
        super().__init__()
        self.input_resolution = input_resolution
        self.proj = nn.Sequential(
            nn.Conv2d(in_channel, in_channel, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(in_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channel, num_class, kernel_size=kernel_size, stride=stride, padding=1),
        )


    def forward(self, x):
        B, L, C = x.shape
        H, W = self.input_resolution
        x = x.transpose(1, 2).view(B, C, H, W)
        x = F.interpolate(x, scale_factor=2, mode='nearest')#for 1024*512

        x = self.proj(x)

        return x
    
class ConvCompressH(nn.Module):
    ''' Reduce feature height by factor of two '''
    def __init__(self, in_c, out_c, ks=3):
        super(ConvCompressH, self).__init__()
        assert ks % 2 == 1
        self.layers = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=ks, stride=(2, 1), padding=ks//2),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.layers(x)


class GlobalHeightConv(nn.Module):
    def __init__(self, in_c, out_c):
        super(GlobalHeightConv, self).__init__()
        self.layer = nn.Sequential(
            ConvCompressH(in_c, in_c//2),
            ConvCompressH(in_c//2, in_c//2),
            ConvCompressH(in_c//2, in_c//4),
            ConvCompressH(in_c//4, out_c),
        )

    def forward(self, x, out_w):
        x = self.layer(x)

        assert out_w % x.shape[3] == 0
        factor = out_w // x.shape[3]
        x = torch.cat([x[..., -1:], x, x[..., :1]], 3)
        x = F.interpolate(x, size=(x.shape[2], out_w + 2 * factor), mode='bilinear', align_corners=False)
        x = x[..., factor:-factor]
        return x


class GlobalHeightStage(nn.Module):
    def __init__(self, c1, c2, c3, c4, out_scale=8):
        ''' Process 4 blocks from encoder to single multiscale features '''
        super(GlobalHeightStage, self).__init__()
        self.cs = c1, c2, c3, c4
        self.out_scale = out_scale
        self.ghc_lst = nn.ModuleList([
            GlobalHeightConv(c1, c1//out_scale),
            GlobalHeightConv(c2, c2//out_scale),
            GlobalHeightConv(c3, c3//out_scale),
            GlobalHeightConv(c4, c4//out_scale),
        ])

    def forward(self, conv_list, out_w):
        assert len(conv_list) == 4
        bs = conv_list[0].shape[0]
        feature = torch.cat([
            f(x, out_w).reshape(bs, -1, out_w)
            for f, x, out_c in zip(self.ghc_lst, conv_list, self.cs)
        ], dim=1)
        return feature



class RGCNet(nn.Module):

    x_mean = torch.FloatTensor(np.array([0.485, 0.456, 0.406])[None, :, None, None])
    x_std = torch.FloatTensor(np.array([0.229, 0.224, 0.225])[None, :, None, None])

    def __init__(self, img_size=256, in_chans=3,
                 embed_dim=32, depths=[2, 2, 2, 2, 2, 2, 2, 2, 2], num_heads=[1, 2, 4, 8, 16, 16, 8, 4, 2],
                 win_size=8, mlp_ratio=4., 
                 drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, patch_norm=True,
                 use_checkpoint=False,
                 dowsample=Downsample, upsample=Upsample, **kwargs):
        super().__init__()

        self.num_enc_layers = len(depths) // 2
        self.num_dec_layers = len(depths) // 2
        self.embed_dim = embed_dim
        self.patch_norm = patch_norm
        self.mlp_ratio = mlp_ratio
        self.win_size = win_size
        self.img_size = img_size
        self.ref_point256x512 = genSamplingPattern(256, 512, 3, 3).cuda()
        self.ref_point128x256 = genSamplingPattern(128, 256, 3, 3).cuda()
        self.ref_point64x128 = genSamplingPattern(64, 128, 3, 3).cuda()
        self.ref_point32x64 = genSamplingPattern(32, 64, 3, 3).cuda()
        self.ref_point16x32 = genSamplingPattern(16, 32, 3, 3).cuda()

        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        enc_dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths[:self.num_enc_layers]))]
        conv_dpr = [drop_path_rate] * depths[4]
        dec_dpr = enc_dpr[::-1]

        # build layers

        # Input projection
        self.input_proj  = InputProj(in_channel=in_chans, out_channel=embed_dim, kernel_size=3, stride=2,
                                    act_layer=nn.GELU)#stride = 2 for 1024*512
        # Output projection
        self.output_proj = OutputProj(in_channel=2 * embed_dim, out_channel=1, kernel_size=3, stride=1,
                                      input_resolution=(img_size, img_size * 2))
        # background segmentation
        self.segmenter   = Segmenter(in_channel =2 * embed_dim, num_class=1, kernel_size=3, stride=1, input_resolution=(img_size, img_size * 2))

        # Encoder
        self.encoderlayer_0 = BasicPanoformerLayer(dim=embed_dim,
                                                output_dim=embed_dim,
                                                input_resolution=(img_size, img_size * 2),
                                                depth=depths[0],
                                                num_heads=num_heads[0],
                                                win_size=win_size,
                                                mlp_ratio=self.mlp_ratio,
                                                drop_path=enc_dpr[int(sum(depths[:0])):int(sum(depths[:1]))],
                                                norm_layer=norm_layer,
                                                use_checkpoint=use_checkpoint,
                                                ref_point=self.ref_point256x512)
        self.dowsample_0 = dowsample(embed_dim, embed_dim * 2, input_resolution=(img_size, img_size * 2))
        self.encoderlayer_1 = BasicPanoformerLayer(dim=embed_dim * 2,
                                                output_dim=embed_dim * 2,
                                                input_resolution=(img_size // 2, img_size * 2 // 2),
                                                depth=depths[1],
                                                num_heads=num_heads[1],
                                                win_size=win_size,
                                                mlp_ratio=self.mlp_ratio,
                                                drop_path=enc_dpr[sum(depths[:1]):sum(depths[:2])],
                                                norm_layer=norm_layer,
                                                use_checkpoint=use_checkpoint,
                                                ref_point=self.ref_point128x256)
        self.dowsample_1 = dowsample(embed_dim * 2, embed_dim * 4, input_resolution=(img_size // 2, img_size * 2 // 2))
        self.encoderlayer_2 = BasicPanoformerLayer(dim=embed_dim * 4,
                                                output_dim=embed_dim * 4,
                                                input_resolution=(img_size // (2 ** 2), img_size * 2 // (2 ** 2)),
                                                depth=depths[2],
                                                num_heads=num_heads[2],
                                                win_size=win_size,
                                                mlp_ratio=self.mlp_ratio,
                                                drop_path=enc_dpr[sum(depths[:2]):sum(depths[:3])],
                                                norm_layer=norm_layer,
                                                use_checkpoint=use_checkpoint,
                                                ref_point=self.ref_point64x128)
        self.dowsample_2 = dowsample(embed_dim * 4, embed_dim * 8,
                                     input_resolution=(img_size // (2 ** 2), img_size * 2 // (2 ** 2)))
        self.encoderlayer_3 = BasicPanoformerLayer(dim=embed_dim * 8,
                                                output_dim=embed_dim * 8,
                                                input_resolution=(img_size // (2 ** 3), img_size * 2 // (2 ** 3)),
                                                depth=depths[3],
                                                num_heads=num_heads[3],
                                                win_size=win_size,
                                                mlp_ratio=self.mlp_ratio,
                                                drop_path=enc_dpr[sum(depths[:3]):sum(depths[:4])],
                                                norm_layer=norm_layer,
                                                use_checkpoint=use_checkpoint,
                                                ref_point=self.ref_point32x64)
        self.dowsample_3 = dowsample(embed_dim * 8, embed_dim * 16,
                                     input_resolution=(img_size // (2 ** 3), img_size * 2 // (2 ** 3)))

        # Bottleneck
        self.conv = BasicPanoformerLayer(dim=embed_dim * 16,
                                      output_dim=embed_dim * 16,
                                      input_resolution=(img_size // (2 ** 4), img_size * 2 // (2 ** 4)),
                                      depth=depths[4],
                                      num_heads=num_heads[4],
                                      win_size=win_size,
                                      mlp_ratio=self.mlp_ratio,
                                      drop_path=conv_dpr,
                                      norm_layer=norm_layer,
                                      use_checkpoint=use_checkpoint,
                                      ref_point=self.ref_point16x32)

        # Decoder
        self.upsample_0 = upsample(embed_dim * 16, embed_dim * 8,
                                   input_resolution=(img_size // (2 ** 4), img_size * 2 // (2 ** 4)))
        self.decoderlayer_0 = BasicPanoformerLayer(dim=embed_dim * 16,
                                                output_dim=embed_dim * 16,
                                                input_resolution=(img_size // (2 ** 3), img_size * 2 // (2 ** 3)),
                                                depth=depths[5],
                                                num_heads=num_heads[5],
                                                win_size=win_size,
                                                mlp_ratio=self.mlp_ratio,
                                                drop_path=dec_dpr[:depths[5]],
                                                norm_layer=norm_layer,
                                                use_checkpoint=use_checkpoint,
                                                ref_point=self.ref_point32x64)
        self.upsample_1 = upsample(embed_dim * 16, embed_dim * 4,
                                   input_resolution=(img_size // (2 ** 3), img_size * 2 // (2 ** 3)))
        self.decoderlayer_1 = BasicPanoformerLayer(dim=embed_dim * 8,
                                                output_dim=embed_dim * 8,
                                                input_resolution=(img_size // (2 ** 2), img_size * 2 // (2 ** 2)),
                                                depth=depths[6],
                                                num_heads=num_heads[6],
                                                win_size=win_size,
                                                mlp_ratio=self.mlp_ratio,
                                                drop_path=dec_dpr[sum(depths[5:6]):sum(depths[5:7])],
                                                norm_layer=norm_layer,
                                                use_checkpoint=use_checkpoint,
                                                ref_point=self.ref_point64x128)
        self.upsample_2 = upsample(embed_dim * 8, embed_dim * 2,
                                   input_resolution=(img_size // (2 ** 2), img_size * 2 // (2 ** 2)))
        self.decoderlayer_2 = BasicPanoformerLayer(dim=embed_dim * 4,
                                                output_dim=embed_dim * 4,
                                                input_resolution=(img_size // 2, img_size * 2 // 2),
                                                depth=depths[7],
                                                num_heads=num_heads[7],
                                                win_size=win_size,
                                                mlp_ratio=self.mlp_ratio,
                                                drop_path=dec_dpr[sum(depths[5:7]):sum(depths[5:8])],
                                                norm_layer=norm_layer,
                                                use_checkpoint=use_checkpoint,
                                                ref_point=self.ref_point128x256)
        self.upsample_3 = upsample(embed_dim * 4, embed_dim, input_resolution=(img_size // 2, img_size * 2 // 2))
        self.decoderlayer_3 = BasicPanoformerLayer(dim=embed_dim * 2,
                                                output_dim=embed_dim * 2,
                                                input_resolution=(img_size, img_size * 2),
                                                depth=depths[8],
                                                num_heads=num_heads[8],
                                                win_size=win_size,
                                                mlp_ratio=self.mlp_ratio,
                                                drop_path=dec_dpr[sum(depths[5:8]):sum(depths[5:9])],
                                                norm_layer=norm_layer,
                                                use_checkpoint=use_checkpoint,
                                                ref_point=self.ref_point256x512)

        # layout decoder
        c1, c2, c3, c4 = 64, 128, 256, 512
        c_last = c1 + c2 // 2 + c3 // 4 + c4 // 8 

        self.reduce_height_module = GlobalHeightStage(c1, c2, c3, c4, 8)

        self.step_cols = 4
        self.bi_rnn = nn.LSTM(input_size=c_last,
                                  hidden_size=512,
                                  num_layers=2,
                                  dropout=0.5,
                                  batch_first=False,
                                  bidirectional=True)
        self.drop_out = nn.Dropout(0.5)
        self.linear = nn.Linear(in_features=2 * 512,
                                out_features=3 * self.step_cols)
        self.linear.bias.data[0*self.step_cols:1*self.step_cols].fill_(-1)
        self.linear.bias.data[1*self.step_cols:2*self.step_cols].fill_(-0.478)
        self.linear.bias.data[2*self.step_cols:3*self.step_cols].fill_(0.425)
        self.x_mean.requires_grad = False
        self.x_std.requires_grad = False

        self.apply(self._init_weights)

        ###
        # frozen the code of layout branch
        self.freezen_list=[self.bi_rnn,self.input_proj,self.segmenter, self.linear,
                           self.encoderlayer_0,self.encoderlayer_1,self.encoderlayer_2,self.encoderlayer_3,
                           self.dowsample_0,self.dowsample_1,self.dowsample_2,self.dowsample_3]
        for submodel in self.freezen_list:
            for name, param in submodel.named_parameters():
                param.requires_grad = False



    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}

    def extra_repr(self) -> str:
        return f"embed_dim={self.embed_dim}, token_projection={self.token_projection}, token_mlp={self.mlp},win_size={self.win_size}"

    def _prepare_x(self, x):
        if self.x_mean.device != x.device:
            self.x_mean = self.x_mean.to(x.device)
            self.x_std = self.x_std.to(x.device)
        return (x[:, :3] - self.x_mean) / self.x_std

    



    def forward(self, x):
        # Input Projection
        pre_x = self._prepare_x(x)
        y = self.input_proj(pre_x)
        y = self.pos_drop(y)

        # Encoder
        conv0 = self.encoderlayer_0(y)
        pool0 = self.dowsample_0(conv0)
        out_layout0 = pool0.reshape(-1, self.img_size // 2, self.img_size, 64).permute(0, 3, 1, 2)


        conv1 = self.encoderlayer_1(pool0)
        pool1 = self.dowsample_1(conv1)
        out_layout1 = pool1.reshape(-1, self.img_size // 4, self.img_size // 2, 128).permute(0, 3, 1, 2)


        conv2 = self.encoderlayer_2(pool1)
        pool2 = self.dowsample_2(conv2)
        out_layout2 = pool2.reshape(-1, self.img_size // 8, self.img_size // 4, 256).permute(0, 3, 1, 2)



        conv3 = self.encoderlayer_3(pool2)
        pool3 = self.dowsample_3(conv3)
        out_layout3 = pool3.reshape(-1, self.img_size // 16, self.img_size // 8, 512).permute(0, 3, 1, 2)


        # code for layout 
        feature = self.reduce_height_module([out_layout0, out_layout1, out_layout2, out_layout3], pre_x.shape[3]//self.step_cols)

        feature = feature.permute(2, 0, 1)  # [w, b, c*h]
        output, _ = self.bi_rnn(feature)  # [seq_len, b, num_directions * hidden_size]
        output = self.drop_out(output)
        output = self.linear(output)  # [seq_len, b, 3 * step_cols]
        output = output.view(output.shape[0], output.shape[1], 3, self.step_cols)  # [seq_len, b, 3, step_cols]
        output = output.permute(1, 2, 0, 3)  # [b, 3, seq_len, step_cols]
        output = output.contiguous().view(output.shape[0], 3, -1)  # [b, 3, seq_len*step_cols]

        cor = output[:, :1]  # B x 1 x W
        bon = output[:, 1:]  # B x 2 x W

        


        # Bottleneck
        conv4 = self.conv(pool3)

        # Decoder
        up0 = self.upsample_0(conv4)
        deconv0 = torch.cat([up0, conv3], -1)
        deconv0 = self.decoderlayer_0(deconv0)

        up1 = self.upsample_1(deconv0)
        deconv1 = torch.cat([up1, conv2], -1)
        deconv1 = self.decoderlayer_1(deconv1)

        up2 = self.upsample_2(deconv1)
        deconv2 = torch.cat([up2, conv1], -1)
        deconv2 = self.decoderlayer_2(deconv2)

        up3 = self.upsample_3(deconv2)
        deconv3 = torch.cat([up3, conv0], -1)
        deconv3 = self.decoderlayer_3(deconv3)

        # Output Projection
        
        y   = self.output_proj(deconv3)
        sem = self.segmenter(deconv3)
        outputs = {}
        outputs["pred_depth"] = y
        outputs["back_segm"]  = sem
        outputs["corner"]     = cor
        outputs["boundry"]    = bon
        return outputs

