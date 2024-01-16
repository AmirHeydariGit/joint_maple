
import torch
import numpy as np
import parser_ops
import utils
import matplotlib.pyplot as plt

parser = parser_ops.get_parser()
args = parser

if torch.cuda.is_available():
    dev = "cuda:0"
else:
   dev = "cpu"

class data_consistency(): 
    def __init__(self, sens_maps,mask):
        self.shape_list = mask.shape
        self.sens_maps = torch.unsqueeze(sens_maps,-1)    
        self.mask = mask
        self.scalar = torch.tensor(torch.tensor(args.ncol_GLOB * args.nrow_GLOB)**(1/2)).to(dev) 
                       

    def Ehe_op(self, img, mu):
        coil_imgs = torch.unsqueeze(self.sens_maps, 0) * torch.unsqueeze(img, 1)  
        kspace = torch.fft.fftshift(torch.fft.fft2(torch.fft.ifftshift(coil_imgs),dim=(2,3)))/self.scalar 
        masked_kspace = kspace * torch.unsqueeze(self.mask,1)
        image_space_coil_imgs = torch.fft.ifftshift(torch.fft.ifft2(torch.fft.fftshift(masked_kspace),dim=(2,3))) * self.scalar
        image_space_comb = torch.sum(image_space_coil_imgs * torch.unsqueeze(torch.conj(self.sens_maps),0), 1)
        
        ispace = image_space_comb + mu *img
        return ispace
    
    def SSDU_kspace(self, img):
        
        
        coil_imgs = torch.unsqueeze(self.sens_maps, 0) * torch.unsqueeze(img, 1)
        kspace = torch.fft.fftshift(torch.fft.fft2(torch.fft.ifftshift(coil_imgs),dim=(2,3))) 
        kspace = kspace / self.scalar
        masked_kspace = kspace * torch.unsqueeze(self.mask, 1)
        
        return masked_kspace
    
def conj_grad(rhs, sens_maps, mask, mu_param, plot=False):
     
     """
Parameters
----------
input_data : contains tuple of  reg output rhs = E^h*y + mu*z , sens_maps and masks
rhs = batch x nrow x ncol x ncont x 2
sens_maps : coil sensitivity maps ncoil x nrow x ncol
mask : batch x nrow x ncol x ncont
mu : penalty parameter
Encoder : Object instance for performing encoding matrix operations
Returns

!!!!!!!!!!!remember to modify to allow arbitrary batch sizes
-------
data consistency output, batch x nrow x ncol x ncont x 2
"""  
     mu_param = torch.tensor(mu_param + 0.j)
     rhs = torch.view_as_complex(rhs)
     encoder = data_consistency(sens_maps, mask)
     x = torch.zeros_like(rhs)
     r, p = rhs, rhs
     rsold = torch.sum(torch.conj(r) * r).float()
     """
     if 0:
         plt.figure()
         plt.imshow(np.squeeze(np.abs(rhs.cpu().numpy()[...,0])), cmap='gray')
         plt.title('input_to_conj_grad')
         plt.show()
         """
     for i in range(args.CG_Iter):
         Ap = encoder.Ehe_op(p, mu_param)
         """
         if 0:
             plt.figure()
             plt.imshow(np.squeeze(np.abs(Ap.cpu().numpy()[...,0])), cmap='gray')
             plt.title('Ap')
             plt.show()
         """
         alpha = torch.tensor(rsold / (torch.sum(torch.conj(p) * Ap).float()) + 0.j)         
         x = x + alpha * p
         r = r - alpha * Ap
         rsnew = torch.sum(torch.conj(r) * r).float()
         beta = rsnew / rsold
         rsold = rsnew
         
         beta = torch.tensor(beta + 0.j)
         p = r + beta * p
         """
         if plot and i == args.CG_Iter -1 :
             plt.figure()
             plt.imshow(np.squeeze(np.abs(x.cpu().numpy()[...,0])), cmap='gray')
             plt.title('p')
             plt.show()   
     """
     x = torch.view_as_real(x)
         
     return x
    
    
   