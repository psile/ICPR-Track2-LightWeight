import torch
from thop import clever_format, profile
from torchsummary import summary

from net import  LightWeightNetwork
# from proposed_model.TwoStreamV2_ITSDT import STNetwork
# from proposed_model.TwoStreamV7 import STNetwork
from load_param_data import  load_param
import pdb
if __name__ == "__main__":
  

    nb_filter, num_blocks= load_param('two', 'resnet_18')
    input       = torch.randn(1, 1, 512, 512).cuda()
    in_channels = 3
  
    # 需要使用device来指定网络在GPU还是CPU运行
    device  = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    m       = LightWeightNetwork()#(num_classes=1, input_channels=in_channels, block=Res_block, num_blocks=num_blocks, nb_filter=nb_filter).to(device)
    #summary(m, (3, 5,input_shape[0], input_shape[1]))
    
    dummy_input     = torch.randn(1, 1, 256,256).to(device)
    flops, params   = profile(m.to(device), (dummy_input, ), verbose=False)
    #--------------------------------------------------------#
    #   flops * 2是因为profile没有将卷积作为两个operations
    #   有些论文将卷积算乘法、加法两个operations。此时乘2
    #   有些论文只考虑乘法的运算次数，忽略加法。此时不乘2
    #   本代码选择乘2，参考YOLOX。
    #--------------------------------------------------------#
    flops           = flops * 2
    flops, params   = clever_format([flops, params], "%.3f")
    print('Total GFLOPS: %s' % (flops))
    print('Total params: %s' % (params))