import torch
from torch import nn 
from mmcv.cnn.bricks import build_activation_layer, build_norm_layer
from mmcv.cnn.utils.weight_init import trunc_normal_init, constant_init
from ..builder import NETWORKS

class FCModule(nn.Module):
    def __init__(self, in_channels, out_channels, 
                bias='auto', norm_cfg=None,act_cfg=None, inplace=True,
                drop_rate=0.0):
        super().__init__()
        self.with_norm = norm_cfg is not None
        self.with_activation = act_cfg is not None
        self.with_dropout = drop_rate >0 
        if bias == 'auto':
            bias = not self.with_norm
        self.with_bias = bias

        self.fc = nn.Linear(in_channels,
                            out_channels, 
                            bias=bias)
        if self.with_norm:
            self.norm = build_norm_layer(norm_cfg, out_channels) 
        if self.with_activation:
            act_cfg_ = act_cfg.copy()
            # nn.Tanh has no 'inplace' argument
            if act_cfg_['type'] not in [
                'Tanh', 'PReLU', 'Sigmoid', 'HSigmoid', 'Swish', 'GELU'
            ]:
                act_cfg_.setdefault('inplace', inplace)
            self.activate = build_activation_layer(act_cfg_)
        if self.with_dropout:
            self.dropout  = nn.Dropout(drop_rate)
        self._init_weights()

    def _init_weights(self):
        trunc_normal_init(self.fc,mean=0,std=0.01)
        if self.with_norm:
            constant_init(self.norm, 1, bias=0)

    def forward(self, x):
        x = self.fc(x)
        if self.with_norm:
            x = self.norm(x)
        if self.with_activation:
            x = self.activate(x)
        if self.with_dropout:
            x = self.dropout(x)
        return x 

@NETWORKS.register_module()
class MLPNet(nn.Module):
    """MLP Network"""
    def __init__(self, 
                in_channels, 
                out_channels, 
                hidden_layers=[50,30], 
                norm_cfg=dict(type='LayerNorm'),
                act_cfg=dict(type='GeLU'),
                drop_rate=0.0,
                ):
        super().__init__()
        layers = []
        dp_rates=[x.item() for x in torch.linspace(0, drop_rate, len(hidden_layers))] 
        for i,channel in enumerate(hidden_layers):
            in_chans = in_channels if i==0 else hidden_layers[i-1]
            layers.append(
                FCModule(in_chans,channel,
                        norm_cfg=norm_cfg,act_cfg=act_cfg, 
                        drop_rate=dp_rates[i]))
        layers.append(nn.Linear(hidden_layers[-1],out_channels))
        self.layers = nn.Sequential(*layers)

    def forward(self,x):
        return self.layers(x)

