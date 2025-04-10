import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt

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
        for self.epoch in range(epochs):
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
            plt.cla()
            plt.plot(np.linspace(1,self.epoch + 1,self.epoch + 1),tlosslist,'r',label='Training')
            plt.plot(np.linspace(1,self.epoch + 1,self.epoch + 1),vlosslist,'g',label='Validation')
            plt.plot(self.best_epoch + 1,best_val_loss,'b*',label='Best result')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()
            if plotType is not None:
                plt.yscale(plotType)
                plt.pause(.01)  # comment out if live plotting not necessary



        self.tlosslist = tlosslist
        self.vlosslist = vlosslist
        
# 3D U-Net solution
class uNet3D(DLN_Base):
    def __init__(self, dev, inChannels=1, outChannels=1, weight=1., basenumfilt=16, filtsz=(3,3,3),
                 min_pos_weight=1.0, max_pos_weight=100.0, 
                 lambda_dice_final=1.0, lambda_shape_final=0.001):
        super().__init__(dev)

        self.weight = torch.tensor(weight, device=dev)

        # Store hyperparameters for loss function
        self.min_pos_weight = min_pos_weight
        self.max_pos_weight = max_pos_weight
        self.lambda_dice_final = lambda_dice_final
        self.lambda_shape_final = lambda_shape_final
        
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


    def init_weights(self,m):
        if isinstance(m,nn.Conv3d):
            nn.init.xavier_normal_(m.weight)

    def myLoss(self, ypred, y):
        v1 = 1
        v2 = 0.001

        intersection = torch.sum(ypred * y) 

        # Loss components
        f = torch.nn.functional.binary_cross_entropy_with_logits
        
        # Calculate Dice loss with epsilon (1e-9)
        dice = 2 * torch.sum(ypred * y) / (torch.sum(ypred) + torch.sum(y) + 1e-9)
        dice_loss = 1 - dice
        
        # Shape loss (MSE)
        loss_shape = 1 / y.numel() * torch.sum((ypred - y) ** 2)
        
        # Use instance hyperparameters
        min_pos_weight = self.min_pos_weight
        max_pos_weight = self.max_pos_weight
        lambda_dice_final = self.lambda_dice_final
        lambda_shape_final = self.lambda_shape_final
        
        # Warmup schedule
        if self.epoch <= 100:
            # Phase 1: Decrease pos_weight from max_pos_weight to min_pos_weight
            pos_weight = max_pos_weight - (max_pos_weight - min_pos_weight) * (self.epoch / 100)
            lambda_dice = 0.01  # Minimal dice influence initially
            lambda_shape = 0.0  # No shape loss initially
            
        elif self.epoch > 100 and self.epoch <= 200:
            # Phase 2: Keep pos_weight constant, increase Dice and shape loss influence
            pos_weight = min_pos_weight
            lambda_dice = 0.01 + (lambda_dice_final - 0.01) * ((self.epoch - 100) / 100)
            lambda_shape = lambda_shape_final * ((self.epoch - 100) / 100)
            
        else:
            # Phase 3: All components at full strength with specified values
            pos_weight = min_pos_weight
            lambda_dice = lambda_dice_final
            lambda_shape = lambda_shape_final
        
        # Calculate BCE loss with dynamic positive weight
        weighted_pos_weight = self.weight + pos_weight 
        bce_loss = v1 * f(ypred, y, pos_weight=weighted_pos_weight) + v2 * intersection
        
        # Total loss
        loss = bce_loss + (lambda_dice * dice_loss) + (lambda_shape * loss_shape)
        
        # For monitoring
        self.current_pos_weight = pos_weight
        self.current_lambda_dice = lambda_dice
        self.current_lambda_shape = lambda_shape
        
        return loss

    #example of basic augmentation
    def Augment(self,D,y):
        # rotate +/- 90 degrees about the z axis
        Dn = torch.cat((D,torch.flip(torch.swapaxes(D,2,3),[2])),dim=0)
        yn = torch.cat((y,torch.flip(torch.swapaxes(y,2,3),[2])),dim=0)
        Dn = torch.cat((Dn,torch.flip(torch.swapaxes(D,2,3),[3])),dim=0)
        yn = torch.cat((yn,torch.flip(torch.swapaxes(y,2,3),[3])),dim=0)

        return Dn,yn
    
def run_experiments():
    # Parameter values to test
    min_pos_weights = [1, 5, 10]
    lambda_dices = [0.5, 1, 2]
    lambda_shapes = [0, 0.001, 0.1]
    
    # Base configuration - the default values
    base_config = {
        'min_pos_weight': 1.0,
        'lambda_dice_final': 1.0,
        'lambda_shape_final': 0.001
    }
    
    results = []
    best_dice = 0
    best_model = None
    best_model_path = None
    
    def evaluate_model(model_path, config, param_type, param_value):
        # Load the trained model
        model = torch.load(model_path, map_location=dev, weights_only=False)
        model.eval()
        
        # Evaluate on test set
        with torch.no_grad():
            test_D_gpu = test_D.to(dev)
            test_y_gpu = test_y.to(dev)
            ypred = model(test_D_gpu) >= 0.5
            
            # Calculate accuracy
            acc = 100 * torch.sum(test_y_gpu == ypred).item() / torch.prod(torch.tensor(test_y_gpu.size()))
            
            # Calculate Dice coefficient
            dice = 2 * torch.sum(ypred * test_y_gpu).item() / (torch.sum(ypred).item() + torch.sum(test_y_gpu).item())
            
            # Calculate confusion matrix metrics
            true_positive = torch.sum((ypred == 1) & (test_y_gpu == 1)).item()
            false_positive = torch.sum((ypred == 1) & (test_y_gpu == 0)).item()
            true_negative = torch.sum((ypred == 0) & (test_y_gpu == 0)).item()
            false_negative = torch.sum((ypred == 0) & (test_y_gpu == 1)).item()
            
            # Calculate additional metrics
            total_voxels = true_positive + false_positive + true_negative + false_negative
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
    
    # Vary min_pos_weight, keeping others fixed
    print("\n=== Testing different min_pos_weight values ===")
    for mpw in min_pos_weights:
        config = base_config.copy()
        config['min_pos_weight'] = mpw
        model_name = f"ChiasmUNet_mpw{mpw}.pth"
        
        print(f"\nTraining with min_pos_weight={mpw}...")
        model = uNet3D(dev, weight=weight_f, 
                      min_pos_weight=config['min_pos_weight'],
                      lambda_dice_final=config['lambda_dice_final'],
                      lambda_shape_final=config['lambda_shape_final'])
        model.to(dev)
        
        opt = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=3e-5, momentum=.9, nesterov=True)
        model.fit(500, model.myLoss, opt, train_D, train_y, valid_D, valid_y, bs, model_name, plotType='log')
        
        # Evaluate the model
        result, dice = evaluate_model(model_name, config, 'min_pos_weight', mpw)
        results.append(result)
        
        # Track best model
        if dice > best_dice:
            best_dice = dice
            best_model_path = model_name
    
    # Vary lambda_dice, keeping others fixed
    print("\n=== Testing different lambda_dice values ===")
    for ld in lambda_dices:
        config = base_config.copy()
        config['lambda_dice_final'] = ld
        model_name = f"ChiasmUNet_ld{ld}.pth"
        
        print(f"\nTraining with lambda_dice={ld}...")
        model = uNet3D(dev, weight=weight_f,
                      min_pos_weight=config['min_pos_weight'],
                      lambda_dice_final=config['lambda_dice_final'],
                      lambda_shape_final=config['lambda_shape_final'])
        model.to(dev)
        
        opt = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=3e-5, momentum=.9, nesterov=True)
        model.fit(500, model.myLoss, opt, train_D, train_y, valid_D, valid_y, bs, model_name, plotType='log')
        
        # Evaluate the model
        result, dice = evaluate_model(model_name, config, 'lambda_dice', ld)
        results.append(result)
        
        # Track best model
        if dice > best_dice:
            best_dice = dice
            best_model_path = model_name
    
    # Vary lambda_shape, keeping others fixed
    print("\n=== Testing different lambda_shape values ===")
    for ls in lambda_shapes:
        config = base_config.copy()
        config['lambda_shape_final'] = ls
        model_name = f"ChiasmUNet_ls{ls}.pth"
        
        print(f"\nTraining with lambda_shape={ls}...")
        model = uNet3D(dev, weight=weight_f,
                      min_pos_weight=config['min_pos_weight'],
                      lambda_dice_final=config['lambda_dice_final'],
                      lambda_shape_final=config['lambda_shape_final'])
        model.to(dev)
        
        opt = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=3e-5, momentum=.9, nesterov=True)
        model.fit(500, model.myLoss, opt, train_D, train_y, valid_D, valid_y, bs, model_name, plotType='log')
        
        # Evaluate the model
        result, dice = evaluate_model(model_name, config, 'lambda_shape', ls)
        results.append(result)
        
        # Track best model
        if dice > best_dice:
            best_dice = dice
            best_model_path = model_name
    
    # Analyze best model more thoroughly
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
            
            if best_result['bias'] == "False Positives":
                print("The model tends to over-segment, predicting more positive regions than actually exist.")
            elif best_result['bias'] == "False Negatives":
                print("The model tends to under-segment, missing some regions that should be positively labeled.")
            
            # Visualize sample results
            visualize_sample_predictions(best_model, test_D, test_y, num_samples=3)
    
    return best_model_path

def visualize_sample_predictions(model, test_D, test_y, num_samples=3):
    """
    Visualize sample predictions from the model compared to ground truth
    """
    # Select random samples
    indices = np.random.choice(len(test_D), min(num_samples, len(test_D)), replace=False)
    
    model.eval()
    with torch.no_grad():
        for i, idx in enumerate(indices):
            # Get prediction
            input_sample = test_D[idx:idx+1].to(dev)
            true_sample = test_y[idx:idx+1]
            pred_sample = model(input_sample) >= 0.5
            
            # Move to CPU for visualization
            pred_sample = pred_sample.cpu()
            
            # Select a middle slice for visualization
            slice_idx = 3
            
            fig, ax = plt.subplots(1, 3, figsize=(15, 5))
            
            # Input image
            ax[0].imshow(np.squeeze(test_D[idx, 0, :, :, slice_idx]).T, 'gray', vmin=0, vmax=100)
            ax[0].set_title('Input Image')
            ax[0].axis('off')
            
            # Ground truth
            ax[1].imshow(np.squeeze(true_sample[0, 0, :, :, slice_idx]).T, 'gray')
            ax[1].set_title('Ground Truth')
            ax[1].axis('off')
            
            # Prediction
            ax[2].imshow(np.squeeze(pred_sample[0, 0, :, :, slice_idx]).T, 'gray')
            ax[2].set_title('Prediction')
            ax[2].axis('off')
            
            plt.tight_layout()
            plt.savefig(f'sample_prediction_{idx}.png')
            plt.close()
            
            print(f"Sample {idx} visualization saved as sample_prediction_{idx}.png")

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

# check for class imbalance
weight_f = torch.sum(train_y==False)/torch.sum(train_y)
print(f'weight: {weight_f}')
# many more foreground than background voxels

m = uNet3D(dev, weight = weight_f)
m.to(dev)
lr = 1e-2
bs = 50
opt = torch.optim.SGD(m.parameters(), lr=lr, weight_decay=3e-5, momentum=.9, nesterov=True)

run_experiments()




