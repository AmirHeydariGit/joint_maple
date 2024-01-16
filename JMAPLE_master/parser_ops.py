
import argparse

def get_parser():
    parser = argparse.ArgumentParser(description='Joint MAPLE: Accelerated T1 and T*2 parameter mapping')

    # %% paths

    parser.add_argument('--data_dir', type=str, default = './dataset/',
                    help='data directory') 
    parser.add_argument('--model_dir', type=str, default = './model/',
                    help='network weights and model directory')
    parser.add_argument('--recon_dir', type=str, default = './recon/',
                    help='reconstructed memfa images directory')
    parser.add_argument('--param_dir', type=str, default = './parameters/',
                    help='Estimated parameter maps directory')
    parser.add_argument('--golden_dir', type=str, default = './golden_parameters/',
                    help='golden fully-sampled parameter maps directory')
              
    # %% dataset
    
    parser.add_argument('--nrow_GLOB', type=int, default = 224,
                        help='number of rows of the slices in the dataset')
    parser.add_argument('--ncol_GLOB', type=int, default = 192,
                        help='number of columns of the slices in the dataset')
    parser.add_argument('--ncoil_GLOB', type=int, default = 32,
                        help='number of coils of the slices in the dataset')
    parser.add_argument('--nte_GLOB', type=int, default = 6,
                        help='number of echoe times in the dataset')
    parser.add_argument('--nfa_GLOB', type=int, default = 3,
                        help='number of flip angles in the dataset') 
    parser.add_argument('--ncont_GLOB', type=int, default = 18,
                        help='number of contrasts in the dataset (nte*nfa)')
    # %% Joint ZS-SSL network and training
              
    parser.add_argument('--epochs', type=int, default = 0,
                        help='number of epochs to train')
    parser.add_argument('--zs_ssl_lr', type=float, default = 5e-4,
                        help='learning rate for joint zs-ssl reconstruction')
    parser.add_argument('--batchSize', type=int, default = 1,
                        help='batch size')
    parser.add_argument('--nb_unroll_blocks', type=int, default = 8,
                        help='number of unrolled blocks')
    parser.add_argument('--nb_res_blocks', type=int, default = 12,
                        help="number of residual blocks in ResNet")
    parser.add_argument('--CG_Iter', type=int, default = 6,
                        help='number of Conjugate Gradient iterations for DC')
    parser.add_argument('--rho_val', type=float, default = 0.2,
                        help='cardinality of the validation mask')                        
    parser.add_argument('--rho_train', type=float, default = 0.4,
                        help='cardinality of the loss mask')
    parser.add_argument('--num_reps', type=int, default = 15,
                        help='number of repetions for the remainder mask')
    parser.add_argument('--transfer_learning', type=bool, default=False,
                        help='transfer learning from pretrained model')                                           
    parser.add_argument('--stop_training', type=int, default = 12,
                        help='stop training if a new lowest validation loss hasnt been achieved in xx epochs')                     
   

# %% Parameter Estimation

    parser.add_argument('--JM_LR', type=float, default = 1e-4,
                       help='learning rate of joint maple training')
    parser.add_argument('--init_LR', type=float, default = 1e-3,
                       help='learning rate of parameter initialization')
    parser.add_argument('--MU', type=float, default =1e2,
                       help='the weight of Loss2 in total loss term')
    parser.add_argument('--JM_Epochs', type=float, default = 20000,
                       help='epoch number of joint maple training')
    parser.add_argument('--init_Epochs', type=float, default = 10000,
                       help='epoch number of parameter initialization')
    parser.add_argument('--Tol', type=float, default = 1e-6,
                       help='the tolerance value for loss function in joint maple training')
    return parser.parse_args("")