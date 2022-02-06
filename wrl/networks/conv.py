from mmcv.cnn.bricks import ConvModule 
from torch import nn 

class ConvNet(nn.Module):
    def __init__(self, 
                    in_channel, 
                    out_channels, 
                    kernel_sizes, 
                    strides,
                    norm_cfg=None,
                    act_cfg=None,
                    flatten=True
                    ):
        super(ConvNet, self).__init__()
        in_dim = in_channel
        convs = []
        for (out_dim, kernel_size, stride) in zip(out_channels, kernel_sizes,strides):
            conv = ConvModule(
                        in_dim, out_dim, kernel_size, 
                        stride, padding=kernel_size//2,
                        norm_cfg=norm_cfg, act_cfg=act_cfg)
            in_dim = out_dim
            convs.append(conv)
        self.convs = nn.Sequential(*convs)
        self.flatten = flatten
    def forward(self, x):
        x = self.convs(x)
        if self.flatten:
            x = x.flatten(1)
        return x
