import torch
import torch.nn as nn
from models.networks.base_network import BaseNetwork
from models.networks.architecture import ElaINResnetBlock as ElaINResnetBlock

class ElaINGenerator(BaseNetwork):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        return parser

    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        nf = opt.ngf #64
        self.fc = nn.Conv1d(3, 16 * nf, 3, padding=1)

        self.conv1 = torch.nn.Conv1d(16 * nf, 16 * nf, 1) 
        self.conv2 = torch.nn.Conv1d(16 * nf, 8 * nf, 1) 
        self.conv3 = torch.nn.Conv1d(8 * nf, 4 * nf, 1) 
        self.conv4 = torch.nn.Conv1d(4 * nf, 3, 1) 

        self.elain_block1 = ElaINResnetBlock(16 * nf, 16 * nf, 128)
        self.elain_block2 = ElaINResnetBlock(8 * nf, 8 * nf, 128)
        self.elain_block3 = ElaINResnetBlock(4 * nf, 4 * nf, 128)


    def forward(self, identity_features, warp_out):
        x = warp_out.transpose(2,1)
        addition = identity_features.transpose(2,1)

        x = self.fc(x)
        x = self.conv1(x)
        x = self.elain_block1(x, addition)
        x = self.conv2(x)
        x = self.elain_block2(x, addition)
        x = self.conv3(x)
        x = self.elain_block3(x, addition)        
        x = 2*torch.tanh(self.conv4(x))

        return x
