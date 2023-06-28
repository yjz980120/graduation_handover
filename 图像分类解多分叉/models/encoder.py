import torch
import torch.nn as nn
import copy 
import numpy as np

from neuronet.models.base_model import BaseModel
from neuronet.models.modules.modules import ConvDropoutNormNonlin, InitWeights_He, StackedConvLayers

class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, down_kernel=(3,3,3), down_stride=(2,2,2)):
        super(DownBlock, self).__init__()
        padding = tuple((k-1)//2 for k in down_kernel)
        down_kwargs = {
            'kernel_size': down_kernel,
            'stride': down_stride,
            'padding': padding,
            'dilation': 1,
            'bias': True,
        }
        self.down = ConvDropoutNormNonlin(in_channels, out_channels, conv_kwargs=down_kwargs)
        conv_kwargs = copy.deepcopy(down_kwargs)
        conv_kwargs['stride'] = 1
        self.conv = ConvDropoutNormNonlin(out_channels, out_channels, conv_kwargs=conv_kwargs)

    def forward(self, x):
        x = self.down(x)
        x = self.conv(x)
        
        return x

class UNet(BaseModel):
    MAX_NUM_FILTERS_3D = 320

    def __init__(self, in_channels, base_num_filters, class_num, down_kernel_list, stride_list, output_bias=False):
        super(UNet, self).__init__()
        assert len(down_kernel_list) == len(stride_list)
        self.downs = []
        # if direct_supervision turned on, no nonlinear or reprojection
        # for the highest resolution image, and the side loss must
        # add additional nonlinear reprojection.
        # self.direct_supervision = direct_supervision

        # the first layer to process the input image
        self.pre_layer = nn.Sequential(
            ConvDropoutNormNonlin(in_channels, base_num_filters),
            ConvDropoutNormNonlin(base_num_filters,base_num_filters),
            )

        in_channels = base_num_filters
        out_channels = 2 * base_num_filters
        down_filters = []
        for i in range(len(down_kernel_list)):
            down_kernel = down_kernel_list[i]
            stride = stride_list[i]
            down_filters.append((in_channels,out_channels))
            down = DownBlock(in_channels, out_channels, down_kernel=down_kernel, down_stride=stride)
            self.downs.append(down)
            in_channels = min(out_channels, self.MAX_NUM_FILTERS_3D)
            if i == 0:
                out_channels = min(out_channels, self.MAX_NUM_FILTERS_3D)
            else:
                out_channels = min(out_channels * 2, self.MAX_NUM_FILTERS_3D)
            
        in_channels = down_filters[-1][-1]
        self.bottleneck = ConvDropoutNormNonlin(in_channels, in_channels)

        final_channels = in_channels * 2 * 2 * 2
        self.fcs = []
        num_projs = 1 
        for _ in range(num_projs):
            #self.fcs.append(ConvDropoutNormNonlin(final_channels,final_channels//2,conv_kwargs={'kernel_size': 1, 'stride': 1, 'padding': 0, 'dilation': 1, 'bias': False},dropout_op_kwargs={'p': 0.5, 'inplace':True}))
            #final_channels //= 2
            self.fcs.append(nn.Sequential(
                nn.Linear(final_channels, final_channels//2),
                nn.Dropout(p=0.5, inplace=True),
                nn.BatchNorm1d(final_channels//2), 
                nn.LeakyReLU(negative_slope=0.01, inplace=True)))
            final_channels = final_channels // 2
        
        self.class_conv = nn.Linear(final_channels, class_num)
        self.downs = nn.ModuleList(self.downs)
        self.fcs = nn.ModuleList(self.fcs)

    def forward(self, x):
        #import ipdb; ipdb.set_trace()
        x = self.pre_layer(x)
        ndown = len(self.downs)
        for i in range(ndown):
            x = self.downs[i](x)
        x = self.bottleneck(x)
        x = x.view(x.shape[0],-1)
        for j in range(len(self.fcs)):
            x = self.fcs[j](x)
        
        class_output = self.class_conv(x)
        return class_output


if __name__ == '__main__':
    import json
    from torchinfo import summary
    #import ipdb; ipdb.set_trace()

    
    """
    in_channels = 7
    base_num_filters = 8
    class_num = 6
    down_kernel_list = [[3,3,3], [3,3,3], [3,3,3], [3,3,3]]
    stride_list = [[2,2,2], [2,2,2], [2,2,2], [2,2,2]]
    output_bias = False
    """    

    conf_file = 'configs/default_config.json'
    with open(conf_file) as fp:
        configs = json.load(fp)
    print("Initialize model...")

    input = torch.randn(2, configs["in_channels"], 32,32,32)
    model = UNet(**configs)
    print(model)

    output = model(input)
    print("output size: ", output.size())
    print("output: ", output)

    summary(model,input_size=(2,configs["in_channels"],32,32,32))


 
