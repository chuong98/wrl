import torch
import numpy as np
from torch import flatten, nn 
from mmcv.cnn.bricks import build_activation_layer, build_norm_layer
from mmcv.cnn.utils.weight_init import trunc_normal_init, constant_init
from ..builder import NETWORKS
from typing import Any, Dict, List, Optional, Sequence, Tuple, Type, Union

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
            self.norm_name, norm = build_norm_layer(norm_cfg, out_channels) 
            self.add_module(self.norm_name, norm)
        else:
            self.norm_name=None
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

    @property
    def norm(self):
        if self.norm_name:
            return getattr(self, self.norm_name)
        else:
            return None

    def forward(self, x):
        x = self.fc(x)
        if self.with_norm:
            x = self.norm(x)
        if self.with_activation:
            x = self.activate(x)
        if self.with_dropout:
            x = self.dropout(x)
        return x 


class MLP(nn.Module):
    """MLP Network"""
    def __init__(self, 
                in_channels: int, 
                out_channels: int, 
                hidden_layers: Sequence[int], 
                norm_cfg=dict(type='LN'),
                act_cfg=dict(type='GELU'),
                drop_rate=0.0,
                device: Optional[Union[str, int, torch.device]] = None,
                ):
        super().__init__()
        self.device = device
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

    def forward(
        self,
        obs: Union[np.ndarray, torch.Tensor],
        ) -> torch.Tensor:
        if self.device is not None:
            obs = torch.as_tensor(
                obs,
                device=self.device,  # type: ignore
                dtype=torch.float32,
            )
        return self.layers(obs.flatten(1))

@NETWORKS.register_module()
class MLPNet(nn.Module):
    def __init__(self, 
        in_channels: int,
        out_channels: int,
        hidden_layers: Sequence[int], 
        norm_cfg=dict(type='LN'),
        act_cfg=dict(type='GELU'),
        drop_rate=0.0,
        device: Optional[Union[str, int, torch.device]] = None,
        softmax= False,
        num_atoms= 1,
        ):
        super().__init__()
        self.device = device
        self.softmax = softmax
        self.num_atoms = num_atoms
        self.model = MLP(in_channels,out_channels*num_atoms,hidden_layers, 
                            norm_cfg=norm_cfg,act_cfg=act_cfg,
                            drop_rate=drop_rate, device=device)

    def forward(
        self,
        obs: Union[np.ndarray, torch.Tensor],
        state: Any = None,
        info: Dict[str, Any] = {},
    ) -> Tuple[torch.Tensor, Any]:
        logits = self.model(obs)
        bsz = logits.shape[0]
        if self.num_atoms > 1:
            logits = logits.view(bsz, -1, self.num_atoms)
        if self.softmax:
            logits = torch.softmax(logits, dim=-1)
        return logits, state