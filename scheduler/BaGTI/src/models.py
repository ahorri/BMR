import torch
import torch.nn as nn
from .constants import *
from .npn import *
import numpy as np
from sys import argv

class energy_50(nn.Module):
    def __init__(self):
        super(energy_50, self).__init__()
        self.name = "energy_50"
        self.find = nn.Sequential(
            nn.Linear(50 * 51, 128),
            nn.Softplus(),
            nn.Linear(128, 128),
            nn.Softplus(),
            nn.Linear(128, 64), 
            nn.Tanhshrink(),
            nn.Linear(64, 1),
            nn.Sigmoid())

    def forward(self, x):
        x = x.flatten()
        x = self.find(x)
        return x
        
class energy_latency31_50(nn.Module):
# class ResNet18(nn.Module):
    def __init__(self, in_channels, resblock, outputs=4):
        super(energy_latency31_50,self).__init__()
        self.name = "energy_latency3_50"
        self.layer0 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )

        self.layer1 = nn.Sequential(
            resblock(64, 64, downsample=False),
            resblock(64, 64, downsample=False)
        )

        self.layer2 = nn.Sequential(
            resblock(64, 128, downsample=True),
            resblock(128, 128, downsample=False)
        )

        self.layer3 = nn.Sequential(
            resblock(128, 256, downsample=True),
            resblock(256, 256, downsample=False)
        )


        self.layer4 = nn.Sequential(
            resblock(256, 512, downsample=True),
            resblock(512, 512, downsample=False)
        )

        self.gap = torch.nn.AdaptiveAvgPool2d(1)
        self.fc = torch.nn.Linear(512, outputs)

    def forward(self, input):
        
        # print('88888888888888888888888 aval',input.size())
        if (input.size()!=([1, 1,50, 53])):
            input=input.detach().numpy()
            input=np.broadcast_to(input, (1,1,50,53))
            input = torch.tensor(input, dtype=torch.float,requires_grad=True).to('cuda:0')
            # input=np.broadcast_to(np.array(input), (1,1,50,53))
            # input = torch.tensor(input, dtype=torch.float)
            # print('88888888888888888888888 new aval',input.size())
        input = self.layer0(input)
         #print('88888888888888888888888888888 layer0',input.size())
        input = self.layer1(input)
        #print('88888888888888888888888888888888888 layer1',input.size())
        input = self.layer2(input)
        #print('888888888888888888888888888888888 layer2',input.size())
        input = self.layer3(input)
        #print('layer3',input.size())
        input = self.layer4(input)
        #print('layer4',input.size())
        input = self.gap(input)
        #print('gap',input.size())
        input = torch.flatten(input)
        #print('flatten',len(input))
        input = self.fc(input)
        #print('bad az FC',input)
        if not('train' in argv[0] and 'train' in argv[2]):
            input = Coeff_energy*input[0] + Coeff_response*input[1]+ Coeff_Bisector*input[2]+Coeff_migrate*input[3]
        return input        



class energy_latency32_50(nn.Module):
# class ResNet18(nn.Module):
    def __init__(self, in_channels, resblock, outputs=4):
        super(energy_latency32_50,self).__init__()
        self.name = "energy_latency32_50"
        self.layer0 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )

        self.layer1 = nn.Sequential(
            resblock(64, 64, downsample=False),
            resblock(64, 64, downsample=False)
        )

        self.layer2 = nn.Sequential(
            resblock(64, 128, downsample=True),
            resblock(128, 128, downsample=False)
        )

        self.layer3 = nn.Sequential(
            resblock(128, 256, downsample=True),
            resblock(256, 256, downsample=False)
        )


        self.layer4 = nn.Sequential(
            resblock(256, 512, downsample=True),
            resblock(512, 512, downsample=False)
        )

        self.gap = torch.nn.AdaptiveAvgPool2d(1)
        self.fc = torch.nn.Linear(512, outputs)

    def forward(self, input):
        
        # print('88888888888888888888888 aval',input.size())
        if (input.size()!=([1, 1,50, 57])):
            input=input.detach().numpy()
            input=np.broadcast_to(input, (1,1,50,57))
            # input = torch.tensor(input, dtype=torch.float,requires_grad=True)
            input = torch.tensor(input, dtype=torch.float,requires_grad=True).to('cuda:0')
            # input=np.broadcast_to(np.array(input), (1,1,50,53))

            # input=np.broadcast_to(np.array(input), (1,1,50,53))
            # input = torch.tensor(input, dtype=torch.float)
            # print('88888888888888888888888 new aval',input.size())
        input = self.layer0(input)
         #print('88888888888888888888888888888 layer0',input.size())
        input = self.layer1(input)
        #print('88888888888888888888888888888888888 layer1',input.size())
        input = self.layer2(input)
        #print('888888888888888888888888888888888 layer2',input.size())
        input = self.layer3(input)
        #print('layer3',input.size())
        input = self.layer4(input)
        #print('layer4',input.size())
        input = self.gap(input)
        #print('gap',input.size())
        input = torch.flatten(input)
        #print('flatten',len(input))
        input = self.fc(input)
        #print('bad az FC',input)
        if not('train' in argv[0] and 'train' in argv[2]):
            input = Coeff_energy*input[0] + Coeff_response*input[1]+ Coeff_Bisector*input[2]+Coeff_migrate*input[3]
        return input        





class energy_latency_50(nn.Module):
    def __init__(self):
        super(energy_latency_50, self).__init__()
        self.name = "energy_latency_50"
        self.find = nn.Sequential(
            nn.Linear(50 * 52, 128),
            nn.Softplus(),
            nn.Linear(128, 128),
            nn.Softplus(),
            nn.Linear(128, 64), 
            nn.Tanhshrink(),
            nn.Linear(64, 2),
            nn.Sigmoid())

    def forward(self, x):
        # print("x1",x)
        # ejra roye GPUuuuuuuuuuuuuuuuuuuuuuuuuu
        # x = x.flatten().to('cuda:0')
        x = x.flatten()
        x = self.find(x)
        # print("x2",x)
        if not('train' in argv[0] and 'train' in argv[2]):
            x = Coeff_Energy*x[0] + Coeff_Latency*x[1]
        return x

class energy_latency_10(nn.Module):
    def __init__(self):
        super(energy_latency_10, self).__init__()
        self.name = "energy_latency_10"
        self.find = nn.Sequential(
            nn.Linear(10 * 12, 128),
            nn.Softplus(),
            nn.Linear(128, 128),
            nn.Softplus(),
            nn.Linear(128, 64), 
            nn.Tanhshrink(),
            nn.Linear(64, 2),
            nn.Sigmoid())

    def forward(self, x):
        x = x.flatten()
        x = self.find(x)
        if not('train' in argv[0] and 'train' in argv[2]):
            x = Coeff_Energy*x[0] + Coeff_Latency*x[1]
        return x

class energy_latency2_10(nn.Module):
    def __init__(self):
        super(energy_latency2_10, self).__init__()
        self.name = "energy_latency2_10"
        self.find = nn.Sequential(
            nn.Linear(10 * 14, 128),
            nn.Softplus(),
            nn.Linear(128, 128),
            nn.Softplus(),
            nn.Linear(128, 64), 
            nn.Tanhshrink(),
            nn.Linear(64, 2),
            nn.Sigmoid())

    def forward(self, x):
        x = x.flatten()
        x = self.find(x)
        if not('train' in argv[0] and 'train' in argv[2]):
            x = Coeff_Energy*x[0] + Coeff_Latency*x[1]
        return x

class energy_latency2_50(nn.Module):
    def __init__(self):
        super(energy_latency2_50, self).__init__()
        self.name = "energy_latency2_50"
        self.find = nn.Sequential(
            nn.Linear(50 * 54, 128),
            nn.Softplus(),
            nn.Linear(128, 128),
            nn.Softplus(),
            nn.Linear(128, 64), 
            nn.Tanhshrink(),
            nn.Linear(64, 2),
            nn.Sigmoid())

    def forward(self, x):
        x = x.flatten()
        x = self.find(x)
        if not('train' in argv[0] and 'train' in argv[2]):
            x = Coeff_Energy*x[0] + Coeff_Latency*x[1]
            #x = Coeff_energy*x[0] + Coeff_response*x[1]+ Coeff_Bisector*x[2]
        return x

class energy_latency3_50(nn.Module):
    def __init__(self):
        super(energy_latency3_50, self).__init__()
        self.name = "energy_latency3_50"
        self.find = nn.Sequential(
            nn.Linear(50 * 53, 128),
            nn.Softplus(),
            nn.Linear(128, 128),
            nn.Softplus(),
            nn.Linear(128, 64), 
            nn.Tanhshrink(),
            nn.Linear(64, 3),
            nn.Sigmoid())

    def forward(self, x):
        x = x.flatten()
        x = self.find(x)
        if not('train' in argv[0] and 'train' in argv[2]):
            x = Coeff_energy*x[0] + Coeff_response*x[1]+ Coeff_Bisector*x[2]
        return x

class energy_latency4_50(nn.Module):
    def __init__(self):
        super(energy_latency4_50, self).__init__()
        self.name = "energy_latency4_50"
        self.find = nn.Sequential(
            nn.Linear(50 * 56, 128),
            nn.Softplus(),
            nn.Linear(128, 128),
            nn.Softplus(),
            nn.Linear(128, 64), 
            nn.Tanhshrink(),
            nn.Linear(64, 3),
            nn.Sigmoid())

    def forward(self, x):
        x = x.flatten()
        x = self.find(x)
        if not('train' in argv[0] and 'train' in argv[2]):
            x = Coeff_energy*x[0] + Coeff_response*x[1]+ Coeff_Bisector*x[2]
        return x

class energy_latency5_50(nn.Module):
    def __init__(self):
        super(energy_latency5_50, self).__init__()
        self.name = "energy_latency5_50"
        self.find = nn.Sequential(
            nn.Linear(50 * 57, 128),
            nn.Softplus(),
            nn.Linear(128, 128),
            nn.Softplus(),
            nn.Linear(128, 64), 
            nn.Tanhshrink(),
            nn.Linear(64, 4),
            nn.Sigmoid())

    def forward(self, x):
        x = x.flatten()
        x = self.find(x)
        if not('train' in argv[0] and 'train' in argv[2]):
            x = Coeff_energy*x[0] + Coeff_response*x[1]+ Coeff_Bisector*x[2]+Coeff_migrate*x[3]
        return x

class energy_latency6_50(nn.Module):
    def __init__(self):
        super(energy_latency6_50, self).__init__()
        self.name = "energy_latency6_50"
        self.find = nn.Sequential(
            nn.Linear(50 * 53, 128),
            nn.Softplus(),
            nn.Linear(128, 128),
            nn.Softplus(),
            nn.Linear(128, 64), 
            nn.Tanhshrink(),
            nn.Linear(64, 4),
            nn.Sigmoid())

    def forward(self, x):
        x = x.flatten()
        x = self.find(x)
        if not('train' in argv[0] and 'train' in argv[2]):
            x = Coeff_energy*x[0] + Coeff_response*x[1]+ Coeff_Bisector*x[2]+Coeff_migrate*x[3]
        return x


class stochastic_energy_latency_50(nn.Module):
    def __init__(self):
        super(stochastic_energy_latency_50, self).__init__()
        self.name = "stochastic_energy_latency_50"
        self.find = nn.Sequential(
            NPNLinear(50 * 52, 128, False),
            NPNRelu(),
            NPNLinear(128, 128),
            NPNRelu(),
            NPNLinear(128, 64), 
            NPNRelu(),
            NPNLinear(64, 1),
            NPNSigmoid())

    def forward(self, x):
        x = x.reshape(1, -1)
        x, s = self.find(x)
        if not('train' in argv[0] and 'train' in argv[2]):
            return x + UCB_K * s
        return x, s

class stochastic_energy_latency2_50(nn.Module):
    def __init__(self):
        super(stochastic_energy_latency2_50, self).__init__()
        self.name = "stochastic_energy_latency2_50"
        self.find = nn.Sequential(
            NPNLinear(50 * 54, 128, False),
            NPNRelu(),
            NPNLinear(128, 128),
            NPNRelu(),
            NPNLinear(128, 64), 
            NPNRelu(),
            NPNLinear(64, 1),
            NPNSigmoid())

    def forward(self, x):
        x = x.reshape(1, -1)
        x, s = self.find(x)
        if not('train' in argv[0] and 'train' in argv[2]):
            return x + UCB_K * s
        return x, s

class stochastic_energy_latency_10(nn.Module):
    def __init__(self):
        super(stochastic_energy_latency_10, self).__init__()
        self.name = "stochastic_energy_latency_10"
        self.find = nn.Sequential(
            NPNLinear(10 * 12, 128, False),
            NPNRelu(),
            NPNLinear(128, 128),
            NPNRelu(),
            NPNLinear(128, 64), 
            NPNRelu(),
            NPNLinear(64, 1),
            NPNSigmoid())

    def forward(self, x):
        x = x.reshape(1, -1)
        x, s = self.find(x)
        if not('train' in argv[0] and 'train' in argv[2]):
            return x + UCB_K * s
        return x, s

class stochastic_energy_latency2_10(nn.Module):
    def __init__(self):
        super(stochastic_energy_latency2_10, self).__init__()
        self.name = "stochastic_energy_latency2_10"
        self.find = nn.Sequential(
            NPNLinear(10 * 14, 128, False),
            NPNRelu(),
            NPNLinear(128, 128),
            NPNRelu(),
            NPNLinear(128, 64), 
            NPNRelu(),
            NPNLinear(64, 1),
            NPNSigmoid())

    def forward(self, x):
        x = x.reshape(1, -1)
        x, s = self.find(x)
        if not('train' in argv[0] and 'train' in argv[2]):
            return x + UCB_K * s
        return x, s



class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, downsample):
        super().__init__()
        if downsample:
            self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1)
          #  print('self.conv_1-resblock::',self.conv_1.size())
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=2),
                nn.BatchNorm2d(out_channels)
            )
           # print('self.shortcut-resblock::',self.self.shortcut.size())
        else:
            self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
          #  print('self.conv1-resblock::',self.conv1.size())
            self.shortcut = nn.Sequential()

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
       # print('self.conv2-resblock::',self.conv2.size())
        self.bn1 = nn.BatchNorm2d(out_channels)
        #print('bn1:',self.bn1.size())
        self.bn2 = nn.BatchNorm2d(out_channels)
        #print('self.bn2:',self.bn2.size())

    def forward(self, input):
        shortcut = self.shortcut(input)
        #print('forward-resb-shortcut:',shortcut.size())
        input = nn.ReLU()(self.bn1(self.conv1(input)))
        #print('forward-resb-bn1:',input.size())
        input = nn.ReLU()(self.bn2(self.conv2(input)))
        #print('forward-resb-bn2:',input.size())
        input = input + shortcut
        #print('forward-resb-input-shortcut:',input.size())
        return nn.ReLU()(input)



act_fn_by = {
    "tanh": nn.Tanh,
    "relu": nn.ReLU,
    "leakyrelu": nn.LeakyReLU,
    "gelu": nn.GELU
}
class InceptionBlock(nn.Module):

    def __init__(self, c_in, c_red : dict, c_out : dict, act_fn):
        """
        Inputs:
            c_in - Number of input feature maps from the previous layers
            c_red - Dictionary with keys "3x3" and "5x5" specifying the output of the dimensionality reducing 1x1 convolutions
            c_out - Dictionary with keys "1x1", "3x3", "5x5", and "max"
            act_fn - Activation class constructor (e.g. nn.ReLU)
        """
        super().__init__()

        # 1x1 convolution branch
        self.conv_1x1 = nn.Sequential(
            nn.Conv2d(c_in, c_out["1x1"], kernel_size=1),
            nn.BatchNorm2d(c_out["1x1"]),
            act_fn()
        )
        print('self.conv_1x1-resblock::',self.conv_1x1.size())
        # 3x3 convolution branch
        self.conv_3x3 = nn.Sequential(
            nn.Conv2d(c_in, c_red["3x3"], kernel_size=1),
            nn.BatchNorm2d(c_red["3x3"]),
            act_fn(),
            nn.Conv2d(c_red["3x3"], c_out["3x3"], kernel_size=3, padding=1),
            nn.BatchNorm2d(c_out["3x3"]),
            act_fn()
        )

        # 5x5 convolution branch
        self.conv_5x5 = nn.Sequential(
            nn.Conv2d(c_in, c_red["5x5"], kernel_size=1),
            nn.BatchNorm2d(c_red["5x5"]),
            act_fn(),
            nn.Conv2d(c_red["5x5"], c_out["5x5"], kernel_size=5, padding=2),
            nn.BatchNorm2d(c_out["5x5"]),
            act_fn()
        )

        # Max-pool branch
        self.max_pool = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, padding=1, stride=1),
            nn.Conv2d(c_in, c_out["max"], kernel_size=1),
            nn.BatchNorm2d(c_out["max"]),
            act_fn()
        )

    def forward(self, x):
        x_1x1 = self.conv_1x1(x)
        x_3x3 = self.conv_3x3(x)
        x_5x5 = self.conv_5x5(x)
        x_max = self.max_pool(x)
        x_out = torch.cat([x_1x1, x_3x3, x_5x5, x_max], dim=1)
        if not('train' in argv[0] and 'train' in argv[2]):
            x_out = Coeff_energy*x_out[0] + Coeff_response*x_out[1]+ Coeff_Bisector*x_out[2]
        print('x_out',x_out)
        return x_out