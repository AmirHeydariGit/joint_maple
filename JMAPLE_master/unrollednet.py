
import torch
from torch import nn
import data_consistency
import parser_ops

parser = parser_ops.get_parser()
args = parser

if torch.cuda.is_available():
    dev = "cuda:0"
else:
   dev = "cpu"



class UnrolledNet(nn.Module):
    """
    Parameters
    ----------
    input_x: batch_size x nrow x ncol x ncont x 2
    sens_maps: batch_size x ncoil x nrow x ncol
    trn_mask: batch_size x nrow x ncol x ncont, used in data consistency units
    loss_mask: batch_size x nrow x ncol x ncont, used to define loss in k-space
    args.nb_unroll_blocks: number of unrolled blocks
    args.nb_res_blocks: number of residual blocks in ResNet
    Returns
    ----------
    x: nw output image
    nw_kspace_output: k-space corresponding nw output at loss mask locations
    x0 : dc output without any regularization.
    all_intermediate_results: all intermediate outputs of regularizer and dc units
    mu: learned penalty parameter
    """
    def __init__(self, sens_maps, trn_mask= torch.ones((1,224,192,18)).to(dev), loss_mask=torch.ones((1,224,192,18)).to(dev)):
        super(UnrolledNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=2*args.ncont_GLOB, out_channels=128, kernel_size=3, padding='same')
        self.conv2 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding='same')
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=2*args.ncont_GLOB, kernel_size=3, padding='same')
        self.resnet_layers = nn.ModuleList()
    
        for k in range(args.nb_res_blocks):
            self.resnet_layers.append(nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding='same'))
        
        self.activate = nn.ReLU()
        self.sens_maps = sens_maps
        self.mu = nn.Parameter(torch.tensor([0.05],requires_grad=True))
        self.trn_mask = trn_mask
        self.loss_mask = loss_mask
        
    def set_loss_mask(self, mask):
        self.loss_mask = mask
    
    def set_trn_mask(self, mask):
        self.trn_mask = mask
    
    
    def forward(self, x, plot_when=0):
        input_x, dc_output = x, x  
        mu_init = torch.tensor([0], dtype=torch.float32).to(dev)
        x0 = data_consistency.conj_grad(x, self.sens_maps, self.trn_mask, mu_init, 0==plot_when)
        
        A=x.shape
        
        for i in range(args.nb_unroll_blocks):
            x=x.contiguous().view((A[0],A[1],A[2],A[3]*A[4]))
            x = x.permute(0, 3,1,2)
            x = x.float()
            x = self.conv1(x)
            first_layer = x
            for j in range(args.nb_res_blocks):
                previous_layer = x
                m = self.resnet_layers[j]
                x = self.activate(m(x))                
                x = m(x)
                x = torch.mul(x, torch.tensor([0.1],dtype=torch.float32).to(dev))
                x = x + previous_layer
                
            rb_output = self.conv2(x)
            temp_output = rb_output + first_layer
            x = self.conv3(temp_output)
            denoiser_output = x
            
            x = x.permute(0, 2,3,1)
            x=x.view((A))
            rhs = input_x + self.mu * x

            x = data_consistency.conj_grad(rhs, self.sens_maps, self.trn_mask,self.mu, 0==plot_when)
            
#           x = x / torch.max(torch.abs(torch.view_as_complex(x)))

            
        encoder = data_consistency.data_consistency(self.sens_maps, self.loss_mask)
        
       
        
        nw_kspace_output = encoder.SSDU_kspace(torch.view_as_complex(x))

        return x, nw_kspace_output, x0
            
                
        
        