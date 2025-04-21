""""
# % U-Net 3D implementation
# % ECE 8396: Medical Image Segmentation
# % Spring 2024
# % Author: Prof. Jack Noble; jack.noble@vanderbilt.edu
# % Modified by: Andre Hucke
# % Date: 2024-04-21
# % Parts of this code were created using AI after many attempts to improve it. All code was reviewed and modified by the author.
"""

import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
import os
from tqdm import tqdm, trange

# If have NVIDIA GPU, see https://pytorch.org/get-started/locally/ for correct CUDA version
# After installing the correct version of CUDA, may need to do something like this:
# >>pip install torch==2.5.0+cu118 torchvision==0.20.0+cu118 torchaudio==2.5.0+cu118 --index-url https://download.pytorch.org/whl/cu118
#   I am using Cuda 11.8 so in the command I use (cu118) for the packages and the pytorch wheel
#   If you use 12.1 it would be cu121

# Base class to inherit training/plotting functionality
class DLN_Base(nn.Module):
    def __init__(self,dev):
        super().__init__()
        self.dev = dev
        self.tlosslist = None
        self.vlosslist = None
        self.epoch = 0
        self.best_epoch = 0

    def loss_batch(self,loss_func,xb,yb,opt=None):
        # xb will be of shape bs x 60 in our first test dataset
        # yb will be bs x 1
        loss = loss_func(self(xb),yb)
        if opt is not None:
            loss.backward()  # perform backpropogation
            opt.step()  # take an optimization step
            opt.zero_grad()  # reset gradients to none

        return loss.item()  # get loss value from gpu

    def Augment(self, xb, yb):
        return xb, yb

    def fit(self,epochs,loss_func,opt,train_D,train_y,valid_D,valid_y,bs,savebest=None,plotType='log'):
        # train_D and train_y are torch tensors sitting on CPU
        # valid_D and valid_y are torch tensors sitting on GPU
        N = train_y.size()[0]
        NB = np.ceil(N / bs).astype(np.longlong)
        tlosslist = []
        vlosslist = []
        best_val_loss = np.inf
        for self.epoch in trange(epochs, desc="Training Progress"):
            self.train() # put model in training mode
            losslist = []
            for i in range(NB):
                start_i = i * bs
                end_i = start_i + bs

                xb, yb = self.Augment(train_D[start_i:end_i].to(self.dev), train_y[start_i:end_i].to(self.dev))
                loss = self.loss_batch(loss_func,xb,yb,opt) # take gradient descent step
                losslist.append(loss)

            self.eval() # put model in validation mode
            with torch.no_grad():
                val_loss = self.loss_batch(loss_func,valid_D,valid_y)

            if val_loss<best_val_loss:
                best_val_loss = val_loss
                self.best_epoch = self.epoch
                if savebest is not None:
                    torch.save(self,savebest)

            tlosslist.append(np.mean(losslist))
            vlosslist.append(val_loss)
            
            # Update tqdm with current losses (no matplotlib plotting during training)
            tqdm.write(f"Epoch {self.epoch+1}/{epochs}, Train loss: {tlosslist[-1]:.4f}, Val loss: {val_loss:.4f}")

        # Save loss history
        self.tlosslist = tlosslist
        self.vlosslist = vlosslist
        
        # Plot losses at the end of training if requested
        if plotType is not None:
            plt.figure(figsize=(10, 6))
            plt.plot(np.linspace(1, epochs, epochs), tlosslist, 'r', label='Training')
            plt.plot(np.linspace(1, epochs, epochs), vlosslist, 'g', label='Validation')
            plt.plot(self.best_epoch + 1, best_val_loss, 'b*', label='Best result')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()
            plt.yscale(plotType)
            plt.title('Training History')
            plt.savefig('training_history.png')
            print(f"Training history plot saved as training_history.png")
            plt.show()
        
# 3D U-Net solution
class uNet3D(DLN_Base):
    def __init__(self,dev,inChannels=1,outChannels=1,weight=1.,basenumfilt=16,filtsz=(3,3,3)):
        super().__init__(dev)

        self.weight = torch.tensor(weight, device=dev)

        self.cv11 = torch.nn.Conv3d(inChannels,basenumfilt,kernel_size=filtsz,stride=(1,1,1),padding=(1,1,1))
        self.cv12 = torch.nn.Conv3d(basenumfilt,basenumfilt,kernel_size=filtsz,stride=(1,1,1),padding=(1,1,1))
        self.cv21 = torch.nn.Conv3d(basenumfilt,basenumfilt * 2,kernel_size=filtsz,stride=(1,1,1),padding=(1,1,1))
        self.cv22 = torch.nn.Conv3d(basenumfilt * 2,basenumfilt * 2,kernel_size=filtsz,stride=(1,1,1),padding=(1,1,1))
        self.cv31 = torch.nn.Conv3d(basenumfilt * 2,basenumfilt * 2,kernel_size=filtsz,stride=(1,1,1),padding=(1,1,1))
        self.cv32 = torch.nn.Conv3d(basenumfilt * 2,basenumfilt * 2,kernel_size=filtsz,stride=(1,1,1),padding=(1,1,1))
        self.cv41 = torch.nn.Conv3d(basenumfilt * 4,basenumfilt * 2,kernel_size=filtsz,stride=(1,1,1),padding=(1,1,1))
        self.cv42 = torch.nn.Conv3d(basenumfilt * 2,basenumfilt * 2,kernel_size=filtsz,stride=(1,1,1),padding=(1,1,1))
        self.cv43 = torch.nn.Conv3d(basenumfilt * 3,basenumfilt * 2,kernel_size=filtsz,stride=(1,1,1),padding=(1,1,1))
        self.cv44 = torch.nn.Conv3d(basenumfilt * 2,basenumfilt * 2,kernel_size=filtsz,stride=(1,1,1),padding=(1,1,1))
        self.cv45 = torch.nn.Conv3d(basenumfilt * 2,outChannels,kernel_size=filtsz,stride=(1,1,1),padding=(1,1,1))

        self.rm1 = torch.tensor(np.zeros(basenumfilt,dtype=np.float32)).to(dev)
        self.rv1 = torch.tensor(np.zeros(basenumfilt,dtype=np.float32)).to(dev)
        self.w1 = torch.tensor(np.ones(basenumfilt,dtype=np.float32)).to(dev)
        self.b1 = torch.tensor(np.zeros(basenumfilt,dtype=np.float32)).to(dev)
        self.rm2 = torch.tensor(np.zeros(basenumfilt,dtype=np.float32)).to(dev)
        self.rv2 = torch.tensor(np.zeros(basenumfilt,dtype=np.float32)).to(dev)
        self.w2 = torch.tensor(np.ones(basenumfilt,dtype=np.float32)).to(dev)
        self.b2 = torch.tensor(np.zeros(basenumfilt,dtype=np.float32)).to(dev)
        self.rm3 = torch.tensor(np.zeros(basenumfilt * 2,dtype=np.float32)).to(dev)
        self.rv3 = torch.tensor(np.zeros(basenumfilt * 2,dtype=np.float32)).to(dev)
        self.w3 = torch.tensor(np.ones(basenumfilt * 2,dtype=np.float32)).to(dev)
        self.b3 = torch.tensor(np.zeros(basenumfilt * 2,dtype=np.float32)).to(dev)
        self.rm4 = torch.tensor(np.zeros(basenumfilt * 2,dtype=np.float32)).to(dev)
        self.rv4 = torch.tensor(np.zeros(basenumfilt * 2,dtype=np.float32)).to(dev)
        self.w4 = torch.tensor(np.ones(basenumfilt * 2,dtype=np.float32)).to(dev)
        self.b4 = torch.tensor(np.zeros(basenumfilt * 2,dtype=np.float32)).to(dev)
        self.rm5 = torch.tensor(np.zeros(basenumfilt * 2,dtype=np.float32)).to(dev)
        self.rv5 = torch.tensor(np.zeros(basenumfilt * 2,dtype=np.float32)).to(dev)
        self.w5 = torch.tensor(np.ones(basenumfilt * 2,dtype=np.float32)).to(dev)
        self.b5 = torch.tensor(np.zeros(basenumfilt * 2,dtype=np.float32)).to(dev)

        self.apply(self.init_weights)  ###

    def forward(self,xb):
        stride = (2,2,2)
        reluscale = .2
        m = nn.LeakyReLU(reluscale)
        mp = nn.MaxPool3d(kernel_size=2,stride=stride)
        us = nn.Upsample(scale_factor=2,mode='trilinear')  #
        bn = nn.functional.batch_norm


        xb = m(bn(self.cv11(xb),running_mean=self.rm1,running_var=self.rv1,weight=self.w1,bias=self.b1,
                  training=self.training))
        xb = m(self.cv12(xb))
        xb2 = bn(mp(xb),running_mean=self.rm2,running_var=self.rv2,weight=self.w2,bias=self.b2,training=self.training)
        xb2 = m(bn(self.cv21(xb2),running_mean=self.rm3,running_var=self.rv3,weight=self.w3,bias=self.b3,
                   training=self.training))
        xb2 = m(self.cv22(xb2))

        # bottleneck
        xb3 = bn(mp(xb2),running_mean=self.rm4,running_var=self.rv4,weight=self.w4,bias=self.b4,training=self.training)
        xb3 = m(bn(self.cv31(xb3),running_mean=self.rm5,running_var=self.rv5,weight=self.w5,bias=self.b5,
                   training=self.training))
        xb3 = m(self.cv32(xb3))

        xb3 = us(xb3)
        xb4 = torch.cat((xb3,xb2),dim=1)  ###
        xb4 = m(self.cv41(xb4))
        xb4 = m(self.cv42(xb4))  ###
        xb4 = us(xb4)
        xb5 = torch.cat((xb4,xb),dim=1)  ###
        xb5 = m(self.cv43(xb5))
        xb6 = m(self.cv44(xb5))  ###
        xb6 = self.cv45(xb6)

        sm = torch.nn.Sigmoid()
        return sm(xb6)


    # Add to UNet3D class
    def init_weights(self, m):
        if isinstance(m, nn.Conv3d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
        elif isinstance(m, nn.BatchNorm3d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

    def myLoss(self,ypred,y):
        v1 = 1
        v2 = 0.001
        f = torch.nn.functional.binary_cross_entropy_with_logits
        intersection = -torch.sum(ypred * y)  # autograd compatible
        # loss = v1 * f(ypred,y,pos_weight=self.weight) #+ v2 * intersection
        # intersection.backward()
        vol = torch.mean(torch.sum(ypred, dim=[2,3,4])) # autograd compatible; could do self supervision comparing to target volume
        # in the project you will implement self supervision comparing the predictions to the mean shape of the chiasm in the training dataset
        # example of scheduling the BCE positive class weight
        if self.epoch<80:
            bcew = torch.tensor([100 - self.epoch], device=self.dev)
        else:
            bcew = torch.tensor([20.], device=self.dev)
        loss = v1 * f(ypred,y,pos_weight=self.weight)  + v2 * intersection
        return loss

    #example of basic augmentation
    def Augment(self,D,y):
        # rotate +/- 90 degrees about the z axis
        Dn = torch.cat((D,torch.flip(torch.swapaxes(D,2,3),[2])),dim=0)
        yn = torch.cat((y,torch.flip(torch.swapaxes(y,2,3),[2])),dim=0)
        Dn = torch.cat((Dn,torch.flip(torch.swapaxes(D,2,3),[3])),dim=0)
        yn = torch.cat((yn,torch.flip(torch.swapaxes(y,2,3),[3])),dim=0)

        return Dn,yn

    #example of basic augmentation
    # def Augment(self,D,y):
    #     # rotate +/- 90 degrees about the z axis
    #     Dn = torch.cat((D,torch.flip(torch.swapaxes(D,2,3),[2])),dim=0)
    #     yn = torch.cat((y,torch.flip(torch.swapaxes(y,2,3),[2])),dim=0)
    #     Dn = torch.cat((Dn,torch.flip(torch.swapaxes(D,2,3),[3])),dim=0)
    #     yn = torch.cat((yn,torch.flip(torch.swapaxes(y,2,3),[3])),dim=0)

    #     return Dn,yn
    def Augment(self, D, y):
        """
        Implement data augmentation with rotations around the z axis
        """
        # First rotation: +90 degrees about z-axis
        x_rot1 = torch.flip(torch.swapaxes(D, 2, 3), [2])
        y_rot1 = torch.flip(torch.swapaxes(y, 2, 3), [2])
        
        # Second rotation: -90 degrees about z-axis
        x_rot2 = torch.flip(torch.swapaxes(D, 2, 3), [3])
        y_rot2 = torch.flip(torch.swapaxes(y, 2, 3), [3])
        
        # Concatenate original and rotated data
        x_aug = torch.cat((D, x_rot1, x_rot2), dim=0)
        y_aug = torch.cat((y, y_rot1, y_rot2), dim=0)
        
        return x_aug, y_aug
    
def run_experiments(train_D, train_y, valid_D, valid_y, test_D, test_y, 
                    weight_f, dev, lr=1e-2, bs=50, max_epochs=500,
                    save_results=True, results_filename='experiment_results.json'):
    """
    A modular experiment framework for testing different hyperparameter configurations.
    
    Parameters:
    -----------
    train_D, train_y : torch tensors
        Training data and labels
    valid_D, valid_y : torch tensors
        Validation data and labels
    test_D, test_y : torch tensors
        Test data and labels
    weight_f : float
        Class weight factor
    dev : str
        Device to run on ('cuda' or 'cpu')
    lr : float
        Learning rate
    bs : int
        Batch size
    max_epochs : int
        Maximum number of epochs to train
    save_results : bool
        Whether to save results to a JSON file
    results_filename : str
        Filename to save results to
        
    Returns:
    --------
    str : Path to the best model
    """
    # Parameter values to test
    experiments = [
        # Min positive weight experiments
        {'param_type': 'min_pos_weight', 'values': [1, 5, 10], 
         'base_config': {'min_pos_weight': 1.0, 'lambda_dice_final': 1.0, 'lambda_shape_final': 0.001}},
        
        # Lambda dice experiments  
        {'param_type': 'lambda_dice_final', 'values': [0.5, 1, 2],
         'base_config': {'min_pos_weight': 1.0, 'lambda_dice_final': 1.0, 'lambda_shape_final': 0.001}},
         
        # Lambda shape experiments
        {'param_type': 'lambda_shape_final', 'values': [0, 0.001, 0.1],
         'base_config': {'min_pos_weight': 1.0, 'lambda_dice_final': 1.0, 'lambda_shape_final': 0.001}}
    ]
    
    results = []
    best_dice = 0
    best_model_path = None
    
    for experiment in experiments:
        param_type = experiment['param_type']
        values = experiment['values']
        base_config = experiment['base_config']
        
        print(f"\n=== Testing different {param_type} values ===")
        
        for value in values:
            config = base_config.copy()
            config[param_type] = value
            model_name = f"ChiasmUNet_{param_type[0:2]}{value}.pth"
            
            print(f"\nTraining with {param_type}={value}...")
            
            # Initialize model with current config
            model = uNet3D(dev, weight=weight_f, 
                         min_pos_weight=config['min_pos_weight'],
                         lambda_dice_final=config['lambda_dice_final'],
                         lambda_shape_final=config['lambda_shape_final'])
            model.to(dev)
            
            # Train model
            opt = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=3e-5, momentum=.9, nesterov=True)
            model.fit(max_epochs, model.myLoss, opt, train_D, train_y, valid_D, valid_y, bs, model_name, plotType='log')
            
            # Evaluate model
            result, dice = evaluate_model(model_name, config, param_type, value, 
                                         test_D, test_y, dev)
            results.append(result)
            
            # Track best model
            if dice > best_dice:
                best_dice = dice
                best_model_path = model_name
    
    # Analyze best model
    if best_model_path:
        print(f"\n=== Best model: {best_model_path} with Dice score: {best_dice:.4f} ===")
        best_model = torch.load(best_model_path, map_location=dev, weights_only=False)
        
        # Find corresponding result
        best_result = next((r for r in results if r['dice'] == best_dice), None)
        if best_result:
            print("Best model configuration:")
            for param, value in best_result['config'].items():
                print(f"  {param}: {value}")
            
            print(f"\nBias analysis: {best_result['bias']}")
            
            # Visualize sample results
            visualize_sample_predictions(best_model, test_D, test_y, num_samples=4, dev=dev, save_dir='results')
    
    return best_model_path

def evaluate_model(model_path, config, param_type, param_value, test_D, test_y, dev):
    """
    Evaluate a trained model on test data
    
    Parameters:
    -----------
    model_path : str
        Path to the trained model
    config : dict
        Model configuration
    param_type : str
        Parameter type being varied
    param_value : float
        Parameter value
    test_D, test_y : torch tensors
        Test data and labels
    dev : str
        Device to run on
        
    Returns:
    --------
    dict : Evaluation results
    float : Dice coefficient
    """
    # Load the trained model
    model = torch.load(model_path, map_location=dev, weights_only=False)
    model.eval()
    
    # Evaluate on test set
    with torch.no_grad():
        test_D_gpu = test_D.to(dev)
        test_y_gpu = test_y.to(dev)
        ypred = model(test_D_gpu) >= 0
        
        # Calculate accuracy
        acc = 100 * torch.sum(test_y_gpu == ypred).item() / torch.prod(torch.tensor(test_y_gpu.size()))
        
        # Calculate Dice coefficient
        smooth = 1e-5
        intersection = torch.sum(ypred * test_y_gpu).item()
        dice = (2 * intersection + smooth) / (torch.sum(ypred).item() + torch.sum(test_y_gpu).item() + smooth)
        
        # Calculate confusion matrix metrics
        true_positive = torch.sum((ypred == 1) & (test_y_gpu == 1)).item()
        false_positive = torch.sum((ypred == 1) & (test_y_gpu == 0)).item()
        true_negative = torch.sum((ypred == 0) & (test_y_gpu == 0)).item()
        false_negative = torch.sum((ypred == 0) & (test_y_gpu == 1)).item()
        
        # Calculate additional metrics
        sensitivity = true_positive / (true_positive + false_negative) if (true_positive + false_negative) > 0 else 0
        specificity = true_negative / (true_negative + false_positive) if (true_negative + false_positive) > 0 else 0
        precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) > 0 else 0
        fp_rate = false_positive / (false_positive + true_negative) if (false_positive + true_negative) > 0 else 0
        fn_rate = false_negative / (false_negative + true_positive) if (false_negative + true_positive) > 0 else 0
        
        # Determine bias
        bias = "False Positives" if fp_rate > fn_rate else "False Negatives" if fn_rate > fp_rate else "No bias"
        
        # Store results
        result = {
            'param_type': param_type,
            'param_value': param_value,
            'config': config.copy(),
            'accuracy': acc,
            'dice': dice,
            'sensitivity': sensitivity,
            'specificity': specificity,
            'precision': precision,
            'fp_rate': fp_rate,
            'fn_rate': fn_rate,
            'bias': bias,
            'confusion_matrix': {
                'TP': true_positive,
                'FP': false_positive,
                'TN': true_negative,
                'FN': false_negative
            },
            'validation_loss': model.vlosslist[-1] if model.vlosslist else None,
            'best_epoch': model.best_epoch
        }
        
        print(f"\nResults for {param_type}={param_value}:")
        print(f"  Accuracy: {acc:.2f}%")
        print(f"  Dice coefficient: {dice:.4f}")
        print(f"  Sensitivity: {sensitivity:.4f}")
        print(f"  Specificity: {specificity:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  FP rate: {fp_rate:.4f}, FN rate: {fn_rate:.4f}")
        print(f"  Bias towards: {bias}")
        print(f"  Best epoch: {model.best_epoch}")
        
        return result, dice

def visualize_sample_predictions(model, test_D, test_y, num_samples=4, dev='cuda', save_dir='.'):
    """
    Visualize sample predictions from the model compared to ground truth
    
    Parameters:
    -----------
    model : uNet3D
        Trained model
    test_D, test_y : torch tensors
        Test data and labels
    num_samples : int
        Number of samples to visualize
    dev : str
        Device to run on
    save_dir : str
        Directory to save visualizations
    """
    # Select random samples
    indices = np.random.choice(len(test_D), min(num_samples, len(test_D)), replace=False)
    
    model.eval()
    with torch.no_grad():
        for i, idx in enumerate(indices):
            # Get prediction
            input_sample = test_D[idx:idx+1].to(dev)
            true_sample = test_y[idx:idx+1]
            pred_sample = model(input_sample) >= 0
            pred_prob = model(input_sample).cpu()  # Raw probability predictions
            
            # Move to CPU for visualization
            pred_sample = pred_sample.cpu()
            
            # Select slices for visualization (middle slice)
            depth = input_sample.shape[-1]
            slice_idx = depth // 2
            
            fig, ax = plt.subplots(2, 2, figsize=(10, 10))
            
            # Input image
            ax[0, 0].imshow(np.squeeze(test_D[idx, 0, :, :, slice_idx]).T, 'gray', vmin=0, vmax=100)
            ax[0, 0].set_title('Input Image')
            ax[0, 0].axis('off')
            
            # Ground truth
            ax[0, 1].imshow(np.squeeze(true_sample[0, 0, :, :, slice_idx]).T, 'gray')
            ax[0, 1].set_title('Ground Truth')
            ax[0, 1].axis('off')
            
            # Prediction (binary)
            ax[1, 0].imshow(np.squeeze(pred_sample[0, 0, :, :, slice_idx]).T, 'gray')
            ax[1, 0].set_title('Prediction (Binary)')
            ax[1, 0].axis('off')
            
            # Prediction (probability)
            prob_img = ax[1, 1].imshow(np.squeeze(pred_prob[0, 0, :, :, slice_idx]).T, 'viridis')
            ax[1, 1].set_title('Prediction (Probability)')
            ax[1, 1].axis('off')
            
            # Add colorbar
            cbar = fig.colorbar(prob_img, ax=ax[1, 1], fraction=0.046, pad=0.04)
            cbar.set_label('Probability')
            
            plt.tight_layout()
            plt.savefig(f'sample_prediction_{i}.png')
            plt.close()
            
            print(f"Sample visualization saved as sample_prediction_{i}.png")

if __name__ == "__main__":
    # try the UNet on a small Chiasm dataset
    f = open('EECE_395/chiasm.json')
    d = json.load(f)
    f.close()

    dev = 'cuda'

    D = np.array(d['D'], dtype=np.float32)
    y = np.array(d['y'], dtype=np.float32)

    testf = 1/10
    validf = 1/10
    trainf = 1 - testf - validf

    # example of randomized sampling of data into training/validation/test sets
    rng = np.random.default_rng(0)
    rand_ord = np.argsort(rng.uniform(size=np.shape(D)[0])).astype(np.int32)
    train_last = np.ceil(np.shape(D)[0] * trainf).astype(np.int32)
    train_indx = rand_ord[0:train_last]

    valid_last = np.ceil(np.shape(D)[0] * validf).astype(np.int32) + train_last
    valid_indx = rand_ord[train_last:valid_last]

    test_indx = rand_ord[valid_last::]

    train_D = torch.tensor(D[train_indx])
    train_y = torch.tensor(y[train_indx, np.newaxis, :, :, :])
    valid_D = torch.tensor(D[valid_indx], device=dev)
    valid_y = torch.tensor(y[valid_indx, np.newaxis, :, :, :], device=dev)
    test_D = torch.tensor(D[test_indx])
    test_y = torch.tensor(y[test_indx, np.newaxis, :, :, :])

    # Calculate class imbalance weight to handle the foreground/background imbalance
    weight_f = torch.sum(train_y==False)/torch.sum(train_y)
    print(f'Class imbalance - background/foreground ratio: {weight_f}')

    # Simple training approach using dynamic class weighting
    model = uNet3D(dev, weight=weight_f, min_pos_weight=1.0, max_pos_weight=weight_f.item()*1.2)
    model.to(dev)

    # Training parameters
    lr = 1e-2
    bs = 50
    max_epochs = 300

    # Train with SGD optimizer
    # opt = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=3e-5, momentum=.9, nesterov=True)

    # Use our custom loss function with proper weighting
    # print("Starting training with positive class weighting...")
    # model_name = "ChiasmUNet_weighted.pth"
    # model.fit(max_epochs, model.myLoss, opt, train_D, train_y, valid_D, valid_y, bs, model_name, plotType='log')

    # Evaluate the model
    with torch.no_grad():
        model.eval()
        test_D_gpu = test_D.to(dev)
        test_y_gpu = test_y.to(dev)
        ypred = model(test_D_gpu) >= 0
        
        # Calculate Dice coefficient
        smooth = 1e-5
        intersection = torch.sum(ypred * test_y_gpu).item()
        dice = (2 * intersection + smooth) / (torch.sum(ypred).item() + torch.sum(test_y_gpu).item() + smooth)
        
        print(f"Test Dice score: {dice:.4f}")
        
        # Visualize some sample predictions
        visualize_sample_predictions(model, test_D, test_y, num_samples=4, dev=dev)

    # Optionally run the full experiments
    run_experiments(train_D, train_y, valid_D, valid_y, test_D, test_y, weight_f, dev, lr, bs)




