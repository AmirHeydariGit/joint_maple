
import torch
import torch.nn as nn
import numpy as np
import utils

class T1_Signal(nn.Module):
    def __init__(self, map_init, scale):
        super(T1_Signal,self).__init__()
        self.M0 = nn.Parameter(map_init[0]*scale)
        self.R1=nn.Parameter(map_init[1]*scale)
        
        
    def set_constants(self, ang, TR, B1, phase):
        self.ang = ang
        self.TR = TR
        self.B1 = B1
        self.phase = phase
        

    def forward(self):
        ang_use_T1=self.ang[...,0,:]
        M0_pd=torch.view_as_complex(self.M0)
        T1_TR=((1-torch.exp(-self.TR*self.R1))/(1-torch.cos(ang_use_T1*self.B1[...,0,:])*torch.exp(-self.TR*self.R1)))
        img=M0_pd*torch.sin(ang_use_T1*self.B1[...,0,:])*T1_TR*self.phase

        return img

class T2_Signal(nn.Module):
    def __init__(self, map_init, scale):
        super(T2_Signal,self).__init__()
        self.M0 = nn.Parameter(map_init[0]*scale)
        self.R2=nn.Parameter(map_init[2]*scale)
        self.delB = nn.Parameter(map_init[3]*scale)
        
    def set_te(self,TE):
            self.TE = TE

    def forward(self):

        TE_use_T2 = self.TE[...,:,0]
        M0_pd = torch.view_as_complex(self.M0)
        T2star_freq=torch.exp(utils.complexify(-self.R2*TE_use_T2,2*torch.tensor(np.pi)*(self.delB*TE_use_T2)))
        img=M0_pd*T2star_freq

        return img

class Joint_Signal(nn.Module):
    def __init__(self, map_init, scale):
        super(Joint_Signal,self).__init__()
        self.M0 = nn.Parameter(map_init[0]*scale)
        self.R1 = nn.Parameter(map_init[1]*scale)
        self.R2 = nn.Parameter(map_init[2]*scale)
        self.delB = nn.Parameter(map_init[3]*scale)
        
    def set_constants(self, ang, TE, TR, B1, phase):
        self.ang = ang
        self.TE = TE
        self.TR = TR
        self.B1 = B1
        self.phase = phase
    def forward(self):

        img = torch.view_as_complex(self.M0)\
            *torch.exp(utils.complexify(-self.TE*self.R2,2*np.pi*self.delB*self.TE))\
                *torch.sin(self.ang*self.B1)\
                    *((1-torch.exp(-self.TR*self.R1))/(1-torch.cos(self.ang*self.B1)*torch.exp(-self.TR*self.R1)))
        return img

