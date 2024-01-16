
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from numpy import linalg as LA

def uniform_selection(input_data, input_mask, rho=0.2, small_acs_block=(4, 4)):
    Input_data = input_data
    Input_mask = input_mask
    nrow, ncol, ncont = Input_data.shape[0], Input_data.shape[1], Input_data.shape[3]
    Trn_mask = np.zeros((nrow,ncol,ncont))
    Loss_mask=np.zeros((nrow,ncol,ncont))
    for i in range(ncont):
        input_data= Input_data[...,i]   
        input_mask=Input_mask[...,i]
        center_kx = int(find_center_ind(input_data, axes=(1, 2)))
        center_ky = int(find_center_ind(input_data, axes=(0, 2)))

        temp_mask = np.copy(input_mask)
        temp_mask[center_kx - small_acs_block[0] // 2: center_kx + small_acs_block[0] // 2,
        center_ky - small_acs_block[1] // 2: center_ky + small_acs_block[1] // 2] = 0

        pr = np.ndarray.flatten(temp_mask)
        ind = np.random.choice(np.arange(nrow * ncol),
                            size=int(np.count_nonzero(pr) * rho), replace=False, p=pr / np.sum(pr))

        [ind_x, ind_y] = index_flatten2nd(ind, (nrow, ncol))

        loss_mask = np.zeros_like(input_mask)
        loss_mask[ind_x, ind_y] = 1

        trn_mask = input_mask - loss_mask
        Trn_mask[...,i]=trn_mask
        Loss_mask[...,i]=loss_mask
    return Trn_mask, Loss_mask


def getPSNR(ref, recon):
    """
    Measures PSNR between the reference and the reconstructed images
    """

    mse = np.sum(np.square(np.abs(ref - recon))) / ref.size
    psnr = 20 * np.log10(np.abs(ref.max()) / (np.sqrt(mse) + 1e-10))

    return psnr


def fft(ispace, axes=(0, 1), norm=None, unitary_opt=True):
    """
    Parameters
    ----------
    ispace : coil images of size nrow x ncol x ncoil x ncont.
    axes :   The default is (0, 1).
    norm :   The default is None.
    unitary_opt : The default is True.
    Returns
    -------
    transform image space to k-space.
    """

    kspace = np.fft.fftshift(np.fft.fftn(np.fft.ifftshift(ispace, axes=axes), axes=axes, norm=norm), axes=axes)

    if unitary_opt:

        fact = 1

        for axis in axes:
            fact = fact * kspace.shape[axis]

        kspace = kspace / np.sqrt(fact)

    return kspace


def ifft(kspace, axes=(0, 1), norm=None, unitary_opt=True):
    """
    Parameters
    ----------
    ispace : image space of size nrow x ncol x ncoil x ncont.
    axes :   The default is (0, 1).
    norm :   The default is None.
    unitary_opt : The default is True.
    Returns
    -------
    transform k-space to image space.
    """

    ispace = np.fft.ifftshift(np.fft.ifftn(np.fft.fftshift(kspace, axes=axes), axes=axes, norm=norm), axes=axes)

    if unitary_opt:

        fact = 1

        for axis in axes:
            fact = fact * ispace.shape[axis]

        ispace = ispace * np.sqrt(fact)

    return ispace


def norm(tensor, axes=(0, 1, 2), keepdims=True):
    """
    Parameters
    ----------
    tensor : It can be in image space or k-space.
    axes :  The default is (0, 1, 2).
    keepdims : The default is True.
    Returns
    -------
    tensor : applies l2-norm .
    """
    for axis in axes:
        tensor = np.linalg.norm(tensor, axis=axis, keepdims=True)

    if not keepdims: return tensor.squeeze()

    return tensor


def find_center_ind(kspace, axes=(1, 2, 3)):
    """
    Parameters
    ----------
    kspace : nrow x ncol x ncoil.
    axes :  The default is (1, 2, 3).
    Returns
    -------
    the center of the k-space
    """

    center_locs = norm(kspace, axes=axes).squeeze()

    return np.argsort(center_locs)[-1:]


def index_flatten2nd(ind, shape):
    """
    Parameters
    ----------
    ind : 1D vector containing chosen locations.
    shape : shape of the matrix/tensor for mapping ind.
    Returns
    -------
    list of >=2D indices containing non-zero locations
    """

    array = np.zeros(np.prod(shape))
    array[ind] = 1
    ind_nd = np.nonzero(np.reshape(array, shape))

    return [list(ind_nd_ii) for ind_nd_ii in ind_nd]


def sense1(input_kspace, sens_maps, axes=(0, 1)):
    """
    Parameters
    ----------
    input_kspace : nrow x ncol x ncoil x ncont
    sens_maps : nrow x ncol x ncoil
    axes : The default is (0,1).
    Returns
    -------
    sense1 image
    """
    
        
    image_space = ifft(input_kspace, axes=axes, norm=None, unitary_opt=True)    
    #im = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(input_kspace[:,:,0]), axes=axes))
    
    #plt.figure()
    #plt.subplot(1,2,1), plt.imshow(np.abs(im), cmap='gray')
    #plt.subplot(1,2,2), plt.imshow(np.abs(input_kspace[:,:,0]), cmap='gray')
    #plt.show()
    
    Eh_op = np.conj(sens_maps[...,None]) * image_space
    sense1_image = np.sum(Eh_op, axis=axes[-1] + 1)
    
    
    return sense1_image


def complex2real(input_data):
    """
    Parameters
    ----------
    input_data : row x col
    dtype :The default is np.float32.
    Returns
    -------
    output : row x col x 2
    """

    return np.stack((input_data.real, input_data.imag), axis=-1)


def real2complex(input_data):
    """
    Parameters
    ----------
    input_data : row x col x 2
    Returns
    -------
    output : row x col
    """

    return input_data[..., 0] + 1j * input_data[..., 1]

def ft2_np(data):
    nx,ny = data.shape[:2]
    return np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(data),axes=(0,1))) / np.sqrt(nx*ny)

def ift2_np(data):
    nx,ny = data.shape[:2]
    return np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(data),axes=(0,1))) * np.sqrt(nx*ny)

def complexify(real, imag):
    return real + 1j*imag

def MSEc(x, y):
    MSE=nn.MSELoss()
    return MSE(torch.view_as_real(x), torch.view_as_real(y))

def Ac_Combined(x,mask,coil):
    return mask*ft2c(coil * x)

def Ahc(x,mask,coil):
    return (torch.conj(coil)*ift2c(mask*x)).sum(0)

def ft2c(x):
    return torch.fft.fftshift(torch.fft.fft2(torch.fft.ifftshift(x), dim = (-1,-2),norm = "ortho"))

def ift2c(x):
    return torch.fft.fftshift(torch.fft.ifft2(torch.fft.ifftshift(x), dim = (-1,-2),norm = "ortho"))

def Saturate(T_in,up=100,dw=0):
    
    if up==None or dw ==None:
       return T_in
   
    T = np.copy(T_in)
    T[abs(T)>up]=up
    T[T<dw]=dw
    return T

def showImg(img, name='',v_min=0, v_max=8e-4,c_map='gray'):
    plt.axis('off')
#     plt.axis('equal')
    plt.tight_layout()
    if len(name) >0:
        plt.title(name)
    plt.imshow(img, cmap= c_map,vmin = v_min, vmax=v_max)

def display_all(img):
    plt.figure()
    img=np.reshape(img,(img.shape[0],img.shape[1],img.shape[2]*img.shape[3]))
    if img.ndim < 3:
        showImg(img)
    else:
        num_img = img.shape[2]
        num_rows = int(np.floor(np.sqrt(num_img)))
        num_cols = int(np.ceil( num_img / num_rows ))

        for k in range(num_img):
            plt.subplot(num_rows,num_cols, k+1)
            showImg(img[:,:,k])
            if k+1 == num_cols:
                plt.show()
        
def display_maps(M0, T1, T2, delB, mask):

    M0_range = [0, 10]
    T1_range = [0, 2500]
    T2_range = [0, 100]
    delB_range = [-25e-3, 25e-3]

    plt.figure()
    showImg(mask*abs(M0) ,'M0',M0_range[0],M0_range[1], 'gray')
    plt.colorbar()
    plt.show()

    plt.figure()
    showImg(mask*T1,'T1',T1_range[0], T1_range[1],'hot')
    plt.colorbar()
    plt.show()
    
    
    plt.figure()
    showImg(mask*T2,'T2*',T2_range[0], T2_range[1],'hot')
    plt.colorbar()
    plt.show()
    
    plt.figure()
    showImg(mask*delB,'del B',delB_range[0], delB_range[1],'gray')
    plt.colorbar()
    plt.show()
    
def NRMSE(inp, ideal, mask, *args, **kwargs):
    
    up_sat = kwargs.get('up_sat', None)
    dw_sat = kwargs.get('dw_sat', None)
    
    if inp.ndim == 2:
       out = LA.norm(mask*(Saturate(ideal,up_sat,dw_sat) - Saturate(inp,up_sat,dw_sat))) / LA.norm(mask*Saturate(ideal,up_sat,dw_sat))
    
    else:
       out = LA.norm(mask[...,None,None]*(ideal - inp)) / LA.norm(mask[...,None,None]*ideal)
    
    return out    
    
    
    
    
    
    
    
    