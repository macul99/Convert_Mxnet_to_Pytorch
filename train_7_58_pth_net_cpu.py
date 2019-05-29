import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def load_weights(weight_file):
    if weight_file == None:
        return
    try:
        weights_dict = np.load(weight_file).item()
    except:
        weights_dict = np.load(weight_file, encoding='bytes').item()
    return weights_dict
class KitModel(nn.Module):

    
    def __init__(self, weight_file):
        super(KitModel, self).__init__()
        self.weights_dict = load_weights(weight_file)

        self.stem_conv1 = self.__conv(2, weights_dict=self.weights_dict, name='stem_conv1', in_channels=3, out_channels=64, kernel_size=(3, 3), stride=(1, 1), groups=1, bias=False)
        self.stem_bn2 = self.__batch_normalization(2, 'stem_bn2', weights_dict=self.weights_dict, num_features=64, eps=1.9999999494757503e-05, momentum=0.8999999761581421)
        self.stage1_unit1_bn1 = self.__batch_normalization(2, 'stage1_unit1_bn1', weights_dict=self.weights_dict, num_features=64, eps=1.9999999494757503e-05, momentum=0.8999999761581421)
        self.stage1_unit1_convr = self.__conv(2, weights_dict=self.weights_dict, name='stage1_unit1_convr', in_channels=64, out_channels=64, kernel_size=(1, 1), stride=(2, 2), groups=1, bias=False)
        self.stage1_unit1_conv1 = self.__conv(2, weights_dict=self.weights_dict, name='stage1_unit1_conv1', in_channels=64, out_channels=64, kernel_size=(3, 3), stride=(1, 1), groups=1, bias=False)
        self.stage1_unit1_bnr = self.__batch_normalization(2, 'stage1_unit1_bnr', weights_dict=self.weights_dict, num_features=64, eps=1.9999999494757503e-05, momentum=0.8999999761581421)
        self.stage1_unit1_bn2 = self.__batch_normalization(2, 'stage1_unit1_bn2', weights_dict=self.weights_dict, num_features=64, eps=1.9999999494757503e-05, momentum=0.8999999761581421)
        self.stage1_unit1_conv2 = self.__conv(2, weights_dict=self.weights_dict, name='stage1_unit1_conv2', in_channels=64, out_channels=64, kernel_size=(3, 3), stride=(2, 2), groups=1, bias=False)
        self.stage1_unit1_bn4 = self.__batch_normalization(2, 'stage1_unit1_bn4', weights_dict=self.weights_dict, num_features=64, eps=1.9999999494757503e-05, momentum=0.8999999761581421)
        self.stage2_unit1_bn1 = self.__batch_normalization(2, 'stage2_unit1_bn1', weights_dict=self.weights_dict, num_features=64, eps=1.9999999494757503e-05, momentum=0.8999999761581421)
        self.stage2_unit1_convr = self.__conv(2, weights_dict=self.weights_dict, name='stage2_unit1_convr', in_channels=64, out_channels=128, kernel_size=(1, 1), stride=(2, 2), groups=1, bias=False)
        self.stage2_unit1_conv1 = self.__conv(2, weights_dict=self.weights_dict, name='stage2_unit1_conv1', in_channels=64, out_channels=128, kernel_size=(3, 3), stride=(1, 1), groups=1, bias=False)
        self.stage2_unit1_bnr = self.__batch_normalization(2, 'stage2_unit1_bnr', weights_dict=self.weights_dict, num_features=128, eps=1.9999999494757503e-05, momentum=0.8999999761581421)
        self.stage2_unit1_bn2 = self.__batch_normalization(2, 'stage2_unit1_bn2', weights_dict=self.weights_dict, num_features=128, eps=1.9999999494757503e-05, momentum=0.8999999761581421)
        self.stage2_unit1_conv2 = self.__conv(2, weights_dict=self.weights_dict, name='stage2_unit1_conv2', in_channels=128, out_channels=128, kernel_size=(3, 3), stride=(2, 2), groups=1, bias=False)
        self.stage2_unit1_bn4 = self.__batch_normalization(2, 'stage2_unit1_bn4', weights_dict=self.weights_dict, num_features=128, eps=1.9999999494757503e-05, momentum=0.8999999761581421)
        self.stage2_unit2_bn1 = self.__batch_normalization(2, 'stage2_unit2_bn1', weights_dict=self.weights_dict, num_features=128, eps=1.9999999494757503e-05, momentum=0.8999999761581421)
        self.stage2_unit2_conv1 = self.__conv(2, weights_dict=self.weights_dict, name='stage2_unit2_conv1', in_channels=128, out_channels=128, kernel_size=(3, 3), stride=(1, 1), groups=1, bias=False)
        self.stage2_unit2_bn2 = self.__batch_normalization(2, 'stage2_unit2_bn2', weights_dict=self.weights_dict, num_features=128, eps=1.9999999494757503e-05, momentum=0.8999999761581421)
        self.stage2_unit2_conv2 = self.__conv(2, weights_dict=self.weights_dict, name='stage2_unit2_conv2', in_channels=128, out_channels=128, kernel_size=(3, 3), stride=(1, 1), groups=1, bias=False)
        self.stage2_unit2_bn4 = self.__batch_normalization(2, 'stage2_unit2_bn4', weights_dict=self.weights_dict, num_features=128, eps=1.9999999494757503e-05, momentum=0.8999999761581421)
        self.stage3_unit1_bn1 = self.__batch_normalization(2, 'stage3_unit1_bn1', weights_dict=self.weights_dict, num_features=128, eps=1.9999999494757503e-05, momentum=0.8999999761581421)
        self.stage3_unit1_convr = self.__conv(2, weights_dict=self.weights_dict, name='stage3_unit1_convr', in_channels=128, out_channels=256, kernel_size=(1, 1), stride=(2, 2), groups=1, bias=False)
        self.stage3_unit1_conv1 = self.__conv(2, weights_dict=self.weights_dict, name='stage3_unit1_conv1', in_channels=128, out_channels=256, kernel_size=(3, 3), stride=(1, 1), groups=1, bias=False)
        self.stage3_unit1_bnr = self.__batch_normalization(2, 'stage3_unit1_bnr', weights_dict=self.weights_dict, num_features=256, eps=1.9999999494757503e-05, momentum=0.8999999761581421)
        self.stage3_unit1_bn2 = self.__batch_normalization(2, 'stage3_unit1_bn2', weights_dict=self.weights_dict, num_features=256, eps=1.9999999494757503e-05, momentum=0.8999999761581421)
        self.stage3_unit1_conv2 = self.__conv(2, weights_dict=self.weights_dict, name='stage3_unit1_conv2', in_channels=256, out_channels=256, kernel_size=(3, 3), stride=(2, 2), groups=1, bias=False)
        self.stage3_unit1_bn4 = self.__batch_normalization(2, 'stage3_unit1_bn4', weights_dict=self.weights_dict, num_features=256, eps=1.9999999494757503e-05, momentum=0.8999999761581421)
        self.stage3_unit2_bn1 = self.__batch_normalization(2, 'stage3_unit2_bn1', weights_dict=self.weights_dict, num_features=256, eps=1.9999999494757503e-05, momentum=0.8999999761581421)
        self.stage3_unit2_conv1 = self.__conv(2, weights_dict=self.weights_dict, name='stage3_unit2_conv1', in_channels=256, out_channels=256, kernel_size=(3, 3), stride=(1, 1), groups=1, bias=False)
        self.stage3_unit2_bn2 = self.__batch_normalization(2, 'stage3_unit2_bn2', weights_dict=self.weights_dict, num_features=256, eps=1.9999999494757503e-05, momentum=0.8999999761581421)
        self.stage3_unit2_conv2 = self.__conv(2, weights_dict=self.weights_dict, name='stage3_unit2_conv2', in_channels=256, out_channels=256, kernel_size=(3, 3), stride=(1, 1), groups=1, bias=False)
        self.stage3_unit2_bn4 = self.__batch_normalization(2, 'stage3_unit2_bn4', weights_dict=self.weights_dict, num_features=256, eps=1.9999999494757503e-05, momentum=0.8999999761581421)
        self.stage3_unit3_bn1 = self.__batch_normalization(2, 'stage3_unit3_bn1', weights_dict=self.weights_dict, num_features=256, eps=1.9999999494757503e-05, momentum=0.8999999761581421)
        self.stage3_unit3_conv1 = self.__conv(2, weights_dict=self.weights_dict, name='stage3_unit3_conv1', in_channels=256, out_channels=256, kernel_size=(3, 3), stride=(1, 1), groups=1, bias=False)
        self.stage3_unit3_bn2 = self.__batch_normalization(2, 'stage3_unit3_bn2', weights_dict=self.weights_dict, num_features=256, eps=1.9999999494757503e-05, momentum=0.8999999761581421)
        self.stage3_unit3_conv2 = self.__conv(2, weights_dict=self.weights_dict, name='stage3_unit3_conv2', in_channels=256, out_channels=256, kernel_size=(3, 3), stride=(1, 1), groups=1, bias=False)
        self.stage3_unit3_bn4 = self.__batch_normalization(2, 'stage3_unit3_bn4', weights_dict=self.weights_dict, num_features=256, eps=1.9999999494757503e-05, momentum=0.8999999761581421)
        self.stage3_unit4_bn1 = self.__batch_normalization(2, 'stage3_unit4_bn1', weights_dict=self.weights_dict, num_features=256, eps=1.9999999494757503e-05, momentum=0.8999999761581421)
        self.stage3_unit4_conv1 = self.__conv(2, weights_dict=self.weights_dict, name='stage3_unit4_conv1', in_channels=256, out_channels=256, kernel_size=(3, 3), stride=(1, 1), groups=1, bias=False)
        self.stage3_unit4_bn2 = self.__batch_normalization(2, 'stage3_unit4_bn2', weights_dict=self.weights_dict, num_features=256, eps=1.9999999494757503e-05, momentum=0.8999999761581421)
        self.stage3_unit4_conv2 = self.__conv(2, weights_dict=self.weights_dict, name='stage3_unit4_conv2', in_channels=256, out_channels=256, kernel_size=(3, 3), stride=(1, 1), groups=1, bias=False)
        self.stage3_unit4_bn4 = self.__batch_normalization(2, 'stage3_unit4_bn4', weights_dict=self.weights_dict, num_features=256, eps=1.9999999494757503e-05, momentum=0.8999999761581421)
        self.stage3_unit5_bn1 = self.__batch_normalization(2, 'stage3_unit5_bn1', weights_dict=self.weights_dict, num_features=256, eps=1.9999999494757503e-05, momentum=0.8999999761581421)
        self.stage3_unit5_conv1 = self.__conv(2, weights_dict=self.weights_dict, name='stage3_unit5_conv1', in_channels=256, out_channels=256, kernel_size=(3, 3), stride=(1, 1), groups=1, bias=False)
        self.stage3_unit5_bn2 = self.__batch_normalization(2, 'stage3_unit5_bn2', weights_dict=self.weights_dict, num_features=256, eps=1.9999999494757503e-05, momentum=0.8999999761581421)
        self.stage3_unit5_conv2 = self.__conv(2, weights_dict=self.weights_dict, name='stage3_unit5_conv2', in_channels=256, out_channels=256, kernel_size=(3, 3), stride=(1, 1), groups=1, bias=False)
        self.stage3_unit5_bn4 = self.__batch_normalization(2, 'stage3_unit5_bn4', weights_dict=self.weights_dict, num_features=256, eps=1.9999999494757503e-05, momentum=0.8999999761581421)
        self.stage4_unit1_bn1 = self.__batch_normalization(2, 'stage4_unit1_bn1', weights_dict=self.weights_dict, num_features=256, eps=1.9999999494757503e-05, momentum=0.8999999761581421)
        self.stage4_unit1_convr = self.__conv(2, weights_dict=self.weights_dict, name='stage4_unit1_convr', in_channels=256, out_channels=512, kernel_size=(1, 1), stride=(2, 2), groups=1, bias=False)
        self.stage4_unit1_conv1 = self.__conv(2, weights_dict=self.weights_dict, name='stage4_unit1_conv1', in_channels=256, out_channels=512, kernel_size=(3, 3), stride=(1, 1), groups=1, bias=False)
        self.stage4_unit1_bnr = self.__batch_normalization(2, 'stage4_unit1_bnr', weights_dict=self.weights_dict, num_features=512, eps=1.9999999494757503e-05, momentum=0.8999999761581421)
        self.stage4_unit1_bn2 = self.__batch_normalization(2, 'stage4_unit1_bn2', weights_dict=self.weights_dict, num_features=512, eps=1.9999999494757503e-05, momentum=0.8999999761581421)
        self.stage4_unit1_conv2 = self.__conv(2, weights_dict=self.weights_dict, name='stage4_unit1_conv2', in_channels=512, out_channels=512, kernel_size=(3, 3), stride=(2, 2), groups=1, bias=False)
        self.stage4_unit1_bn4 = self.__batch_normalization(2, 'stage4_unit1_bn4', weights_dict=self.weights_dict, num_features=512, eps=1.9999999494757503e-05, momentum=0.8999999761581421)
        self.stage4_unit2_bn1 = self.__batch_normalization(2, 'stage4_unit2_bn1', weights_dict=self.weights_dict, num_features=512, eps=1.9999999494757503e-05, momentum=0.8999999761581421)
        self.stage4_unit2_conv1 = self.__conv(2, weights_dict=self.weights_dict, name='stage4_unit2_conv1', in_channels=512, out_channels=512, kernel_size=(3, 3), stride=(1, 1), groups=1, bias=False)
        self.stage4_unit2_bn2 = self.__batch_normalization(2, 'stage4_unit2_bn2', weights_dict=self.weights_dict, num_features=512, eps=1.9999999494757503e-05, momentum=0.8999999761581421)
        self.stage4_unit2_conv2 = self.__conv(2, weights_dict=self.weights_dict, name='stage4_unit2_conv2', in_channels=512, out_channels=512, kernel_size=(3, 3), stride=(1, 1), groups=1, bias=False)
        self.stage4_unit2_bn4 = self.__batch_normalization(2, 'stage4_unit2_bn4', weights_dict=self.weights_dict, num_features=512, eps=1.9999999494757503e-05, momentum=0.8999999761581421)
        self.out_bn2 = self.__batch_normalization(2, 'out_bn2', weights_dict=self.weights_dict, num_features=512, eps=1.9999999494757503e-05, momentum=0.8999999761581421)
        self.out_conv1 = self.__conv(2, weights_dict=self.weights_dict, name='out_conv1', in_channels=512, out_channels=512, kernel_size=(3, 3), stride=(1, 1), groups=1, bias=False)
        self.out_bn3 = self.__batch_normalization(2, 'out_bn3', weights_dict=self.weights_dict, num_features=512, eps=1.9999999494757503e-05, momentum=0.8999999761581421)
        self.out_fc1 = self.__dense(name = 'out_fc1', weights_dict=self.weights_dict, in_features = 25088, out_features = 512, bias = True)
        self.out_embedding = self.__batch_normalization(0, 'out_embedding', weights_dict=self.weights_dict, num_features=512, eps=1.9999999494757503e-05, momentum=0.8999999761581421)

    def forward(self, x):
        stem_conv1_pad  = F.pad(x, (1, 1, 1, 1))
        stem_conv1      = self.stem_conv1(stem_conv1_pad)
        stem_bn2        = self.stem_bn2(stem_conv1)
        stem_relu1      = F.prelu(stem_bn2, torch.from_numpy(self.weights_dict['stem_relu1']['weights']))
        stage1_unit1_bn1 = self.stage1_unit1_bn1(stem_relu1)
        stage1_unit1_convr = self.stage1_unit1_convr(stem_relu1)
        stage1_unit1_conv1_pad = F.pad(stage1_unit1_bn1, (1, 1, 1, 1))
        stage1_unit1_conv1 = self.stage1_unit1_conv1(stage1_unit1_conv1_pad)
        stage1_unit1_bnr = self.stage1_unit1_bnr(stage1_unit1_convr)
        stage1_unit1_bn2 = self.stage1_unit1_bn2(stage1_unit1_conv1)
        stage1_unit1_prelu2 = F.prelu(stage1_unit1_bn2, torch.from_numpy(self.weights_dict['stage1_unit1_prelu2']['weights']))
        stage1_unit1_conv2_pad = F.pad(stage1_unit1_prelu2, (1, 1, 1, 1))
        stage1_unit1_conv2 = self.stage1_unit1_conv2(stage1_unit1_conv2_pad)
        stage1_unit1_bn4 = self.stage1_unit1_bn4(stage1_unit1_conv2)
        plus0           = stage1_unit1_bn4 + stage1_unit1_bnr
        stage2_unit1_bn1 = self.stage2_unit1_bn1(plus0)
        stage2_unit1_convr = self.stage2_unit1_convr(plus0)
        stage2_unit1_conv1_pad = F.pad(stage2_unit1_bn1, (1, 1, 1, 1))
        stage2_unit1_conv1 = self.stage2_unit1_conv1(stage2_unit1_conv1_pad)
        stage2_unit1_bnr = self.stage2_unit1_bnr(stage2_unit1_convr)
        stage2_unit1_bn2 = self.stage2_unit1_bn2(stage2_unit1_conv1)
        stage2_unit1_prelu2 = F.prelu(stage2_unit1_bn2, torch.from_numpy(self.weights_dict['stage2_unit1_prelu2']['weights']))
        stage2_unit1_conv2_pad = F.pad(stage2_unit1_prelu2, (1, 1, 1, 1))
        stage2_unit1_conv2 = self.stage2_unit1_conv2(stage2_unit1_conv2_pad)
        stage2_unit1_bn4 = self.stage2_unit1_bn4(stage2_unit1_conv2)
        plus1           = stage2_unit1_bn4 + stage2_unit1_bnr
        stage2_unit2_bn1 = self.stage2_unit2_bn1(plus1)
        stage2_unit2_conv1_pad = F.pad(stage2_unit2_bn1, (1, 1, 1, 1))
        stage2_unit2_conv1 = self.stage2_unit2_conv1(stage2_unit2_conv1_pad)
        stage2_unit2_bn2 = self.stage2_unit2_bn2(stage2_unit2_conv1)
        stage2_unit2_prelu2 = F.prelu(stage2_unit2_bn2, torch.from_numpy(self.weights_dict['stage2_unit2_prelu2']['weights']))
        stage2_unit2_conv2_pad = F.pad(stage2_unit2_prelu2, (1, 1, 1, 1))
        stage2_unit2_conv2 = self.stage2_unit2_conv2(stage2_unit2_conv2_pad)
        stage2_unit2_bn4 = self.stage2_unit2_bn4(stage2_unit2_conv2)
        plus2           = stage2_unit2_bn4 + plus1
        stage3_unit1_bn1 = self.stage3_unit1_bn1(plus2)
        stage3_unit1_convr = self.stage3_unit1_convr(plus2)
        stage3_unit1_conv1_pad = F.pad(stage3_unit1_bn1, (1, 1, 1, 1))
        stage3_unit1_conv1 = self.stage3_unit1_conv1(stage3_unit1_conv1_pad)
        stage3_unit1_bnr = self.stage3_unit1_bnr(stage3_unit1_convr)
        stage3_unit1_bn2 = self.stage3_unit1_bn2(stage3_unit1_conv1)
        stage3_unit1_prelu2 = F.prelu(stage3_unit1_bn2, torch.from_numpy(self.weights_dict['stage3_unit1_prelu2']['weights']))
        stage3_unit1_conv2_pad = F.pad(stage3_unit1_prelu2, (1, 1, 1, 1))
        stage3_unit1_conv2 = self.stage3_unit1_conv2(stage3_unit1_conv2_pad)
        stage3_unit1_bn4 = self.stage3_unit1_bn4(stage3_unit1_conv2)
        plus3           = stage3_unit1_bn4 + stage3_unit1_bnr
        stage3_unit2_bn1 = self.stage3_unit2_bn1(plus3)
        stage3_unit2_conv1_pad = F.pad(stage3_unit2_bn1, (1, 1, 1, 1))
        stage3_unit2_conv1 = self.stage3_unit2_conv1(stage3_unit2_conv1_pad)
        stage3_unit2_bn2 = self.stage3_unit2_bn2(stage3_unit2_conv1)
        stage3_unit2_prelu2 = F.prelu(stage3_unit2_bn2, torch.from_numpy(self.weights_dict['stage3_unit2_prelu2']['weights']))
        stage3_unit2_conv2_pad = F.pad(stage3_unit2_prelu2, (1, 1, 1, 1))
        stage3_unit2_conv2 = self.stage3_unit2_conv2(stage3_unit2_conv2_pad)
        stage3_unit2_bn4 = self.stage3_unit2_bn4(stage3_unit2_conv2)
        plus4           = stage3_unit2_bn4 + plus3
        stage3_unit3_bn1 = self.stage3_unit3_bn1(plus4)
        stage3_unit3_conv1_pad = F.pad(stage3_unit3_bn1, (1, 1, 1, 1))
        stage3_unit3_conv1 = self.stage3_unit3_conv1(stage3_unit3_conv1_pad)
        stage3_unit3_bn2 = self.stage3_unit3_bn2(stage3_unit3_conv1)
        stage3_unit3_prelu2 = F.prelu(stage3_unit3_bn2, torch.from_numpy(self.weights_dict['stage3_unit3_prelu2']['weights']))
        stage3_unit3_conv2_pad = F.pad(stage3_unit3_prelu2, (1, 1, 1, 1))
        stage3_unit3_conv2 = self.stage3_unit3_conv2(stage3_unit3_conv2_pad)
        stage3_unit3_bn4 = self.stage3_unit3_bn4(stage3_unit3_conv2)
        plus5           = stage3_unit3_bn4 + plus4
        stage3_unit4_bn1 = self.stage3_unit4_bn1(plus5)
        stage3_unit4_conv1_pad = F.pad(stage3_unit4_bn1, (1, 1, 1, 1))
        stage3_unit4_conv1 = self.stage3_unit4_conv1(stage3_unit4_conv1_pad)
        stage3_unit4_bn2 = self.stage3_unit4_bn2(stage3_unit4_conv1)
        stage3_unit4_prelu2 = F.prelu(stage3_unit4_bn2, torch.from_numpy(self.weights_dict['stage3_unit4_prelu2']['weights']))
        stage3_unit4_conv2_pad = F.pad(stage3_unit4_prelu2, (1, 1, 1, 1))
        stage3_unit4_conv2 = self.stage3_unit4_conv2(stage3_unit4_conv2_pad)
        stage3_unit4_bn4 = self.stage3_unit4_bn4(stage3_unit4_conv2)
        plus6           = stage3_unit4_bn4 + plus5
        stage3_unit5_bn1 = self.stage3_unit5_bn1(plus6)
        stage3_unit5_conv1_pad = F.pad(stage3_unit5_bn1, (1, 1, 1, 1))
        stage3_unit5_conv1 = self.stage3_unit5_conv1(stage3_unit5_conv1_pad)
        stage3_unit5_bn2 = self.stage3_unit5_bn2(stage3_unit5_conv1)
        stage3_unit5_prelu2 = F.prelu(stage3_unit5_bn2, torch.from_numpy(self.weights_dict['stage3_unit5_prelu2']['weights']))
        stage3_unit5_conv2_pad = F.pad(stage3_unit5_prelu2, (1, 1, 1, 1))
        stage3_unit5_conv2 = self.stage3_unit5_conv2(stage3_unit5_conv2_pad)
        stage3_unit5_bn4 = self.stage3_unit5_bn4(stage3_unit5_conv2)
        plus7           = stage3_unit5_bn4 + plus6
        stage4_unit1_bn1 = self.stage4_unit1_bn1(plus7)
        stage4_unit1_convr = self.stage4_unit1_convr(plus7)
        stage4_unit1_conv1_pad = F.pad(stage4_unit1_bn1, (1, 1, 1, 1))
        stage4_unit1_conv1 = self.stage4_unit1_conv1(stage4_unit1_conv1_pad)
        stage4_unit1_bnr = self.stage4_unit1_bnr(stage4_unit1_convr)
        stage4_unit1_bn2 = self.stage4_unit1_bn2(stage4_unit1_conv1)
        stage4_unit1_prelu2 = F.prelu(stage4_unit1_bn2, torch.from_numpy(self.weights_dict['stage4_unit1_prelu2']['weights']))
        stage4_unit1_conv2_pad = F.pad(stage4_unit1_prelu2, (1, 1, 1, 1))
        stage4_unit1_conv2 = self.stage4_unit1_conv2(stage4_unit1_conv2_pad)
        stage4_unit1_bn4 = self.stage4_unit1_bn4(stage4_unit1_conv2)
        plus8           = stage4_unit1_bn4 + stage4_unit1_bnr
        stage4_unit2_bn1 = self.stage4_unit2_bn1(plus8)
        stage4_unit2_conv1_pad = F.pad(stage4_unit2_bn1, (1, 1, 1, 1))
        stage4_unit2_conv1 = self.stage4_unit2_conv1(stage4_unit2_conv1_pad)
        stage4_unit2_bn2 = self.stage4_unit2_bn2(stage4_unit2_conv1)
        stage4_unit2_prelu2 = F.prelu(stage4_unit2_bn2, torch.from_numpy(self.weights_dict['stage4_unit2_prelu2']['weights']))
        stage4_unit2_conv2_pad = F.pad(stage4_unit2_prelu2, (1, 1, 1, 1))
        stage4_unit2_conv2 = self.stage4_unit2_conv2(stage4_unit2_conv2_pad)
        stage4_unit2_bn4 = self.stage4_unit2_bn4(stage4_unit2_conv2)
        plus9           = stage4_unit2_bn4 + plus8
        out_bn2         = self.out_bn2(plus9)
        out_relu2       = F.prelu(out_bn2, torch.from_numpy(self.weights_dict['out_relu2']['weights']))
        out_conv1_pad   = F.pad(out_relu2, (1, 1, 1, 1))
        out_conv1       = self.out_conv1(out_conv1_pad)
        out_bn3         = self.out_bn3(out_conv1)
        out_relu3       = F.relu(out_bn3)
        out_fc1         = self.out_fc1(out_relu3.view(out_relu3.size(0), -1))
        out_embedding   = self.out_embedding(out_fc1)
        return out_embedding


    @staticmethod
    def __conv(dim, weights_dict, name, **kwargs):
        if   dim == 1:  layer = nn.Conv1d(**kwargs)
        elif dim == 2:  layer = nn.Conv2d(**kwargs)
        elif dim == 3:  layer = nn.Conv3d(**kwargs)
        else:           raise NotImplementedError()
        layer.state_dict()['weight'].copy_(torch.from_numpy(weights_dict[name]['weights']))
        if 'bias' in weights_dict[name]:
            layer.state_dict()['bias'].copy_(torch.from_numpy(weights_dict[name]['bias']))
        return layer

    @staticmethod
    def __batch_normalization(dim, name, weights_dict, **kwargs):
        if   dim == 0 or dim == 1:  layer = nn.BatchNorm1d(**kwargs)
        elif dim == 2:  layer = nn.BatchNorm2d(**kwargs)
        elif dim == 3:  layer = nn.BatchNorm3d(**kwargs)
        else:           raise NotImplementedError()
        if 'scale' in weights_dict[name]:
            layer.state_dict()['weight'].copy_(torch.from_numpy(weights_dict[name]['scale']))
        else:
            layer.weight.data.fill_(1)
        if 'bias' in weights_dict[name]:
            layer.state_dict()['bias'].copy_(torch.from_numpy(weights_dict[name]['bias']))
        else:
            layer.bias.data.fill_(0)
        layer.state_dict()['running_mean'].copy_(torch.from_numpy(weights_dict[name]['mean']))
        layer.state_dict()['running_var'].copy_(torch.from_numpy(weights_dict[name]['var']))
        return layer

    @staticmethod
    def __dense(name, weights_dict, **kwargs):
        layer = nn.Linear(**kwargs)
        layer.state_dict()['weight'].copy_(torch.from_numpy(weights_dict[name]['weights']))
        if 'bias' in weights_dict[name]:
            layer.state_dict()['bias'].copy_(torch.from_numpy(weights_dict[name]['bias']))
        return layer

