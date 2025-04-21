import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import os
import json
from Project6 import uNet3D
from tqdm import tqdm, trange

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        
        # Discriminator architecture as specified in the project requirements
        self.conv1 = nn.Conv3d(2, 64, kernel_size=(3, 3, 3), padding=(1, 1, 1), stride=(2, 2, 2))
        self.bn1 = nn.BatchNorm3d(64)
        self.dropout1 = nn.Dropout(0.2)
        
        self.conv2 = nn.Conv3d(64, 128, kernel_size=(3, 3, 3), padding=(1, 1, 1), stride=(2, 2, 2))
        self.bn2 = nn.BatchNorm3d(128)
        self.dropout2 = nn.Dropout(0.2)

        self.conv3 = nn.Conv3d(128, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1), stride=(2, 2, 2))
        self.bn3 = nn.BatchNorm3d(256)
        self.dropout3 = nn.Dropout(0.2)

        self.conv4 = nn.Conv3d(256, 1, kernel_size=(3, 3, 3), padding=(1, 1, 1), stride=(4, 4, 1))
        
        # Initialize weights using Xavier normal method
        self.apply(self._init_weights)
        
    def _init_weights(self, m):
        if isinstance(m, nn.Conv3d):
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, x, y):
        # Concatenate image and segmentation mask as input
        x = torch.cat([x, y], dim=1)
        
        # Apply convolutional layers with LeakyReLU (negative slope 0.2)
        x = F.leaky_relu(self.bn1(self.conv1(x)), negative_slope=0.2)
        x = self.dropout1(x)
        
        x = F.leaky_relu(self.bn2(self.conv2(x)), negative_slope=0.2)
        x = self.dropout2(x)
        
        x = F.leaky_relu(self.bn3(self.conv3(x)), negative_slope=0.2)
        x = self.dropout3(x)
        
        # Final convolution and sigmoid activation
        x = self.conv4(x)
        x = torch.flatten(x, 1)
        return torch.sigmoid(x)

class cGAN(nn.Module):
    def __init__(self, dev):
        super(cGAN, self).__init__()
        self.dev = dev
        
        # Initialize generator using U-Net from Project 6
        self.generator = uNet3D(dev)
        self.generator.to(dev)  # Explicitly move all parameters of the generator to the device
        
        # Initialize discriminator
        self.discriminator = Discriminator().to(dev)
        
        # Tracking variables
        self.epoch = 0
        self.best_epoch = 0
        self.tlosslist = []
        self.vlosslist = []
    
    def forward(self, x):
        # Forward pass just returns generator output
        return self.generator(x)
    
    def check_gradients(self, model, name='model'):
        """
        Helper function to check if gradients exist and their magnitudes
        
        Parameters:
        -----------
        model : torch.nn.Module
            The model to check gradients for
        name : str
            Name identifier for the model in print statements
            
        Returns:
        --------
        dict
            Dictionary with gradient statistics
        """
        total_norm = 0
        max_norm = 0
        zero_grad_params = 0
        total_params = 0
        
        for p in model.parameters():
            if p.grad is not None:
                total_params += 1
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
                max_norm = max(max_norm, param_norm.item())
                if param_norm.item() == 0:
                    zero_grad_params += 1
            else:
                zero_grad_params += 1
        
        total_norm = total_norm ** 0.5
        
        # Return statistics instead of printing them
        return {
            'avg': total_norm,
            'max': max_norm,
            'zero_percent': 100.0 * zero_grad_params / max(1, total_params)
        }
    
    def Augment(self, x_batch, y_batch):
        """
        Implement data augmentation with rotations around the z axis
        """
        # First rotation: +90 degrees about z-axis
        x_rot1 = torch.flip(torch.swapaxes(x_batch, 2, 3), [2])
        y_rot1 = torch.flip(torch.swapaxes(y_batch, 2, 3), [2])
        
        # Second rotation: -90 degrees about z-axis
        x_rot2 = torch.flip(torch.swapaxes(x_batch, 2, 3), [3])
        y_rot2 = torch.flip(torch.swapaxes(y_batch, 2, 3), [3])
        
        # Concatenate original and rotated data
        x_aug = torch.cat((x_batch, x_rot1, x_rot2), dim=0)
        y_aug = torch.cat((y_batch, y_rot1, y_rot2), dim=0)
        
        return x_aug, y_aug

    def visualize_sample_predictions(self, test_D, test_y, num_samples=4, save_dir='sample_predictions'):
        """
        Visualize sample predictions from the model compared to ground truth
        
        Parameters:
        -----------
        test_D, test_y : torch tensors
            Test data and labels
        num_samples : int
            Number of samples to visualize
        save_dir : str
            Directory to save visualizations
        """
        # Create save directory if it doesn't exist
        os.makedirs(save_dir, exist_ok=True)
        
        # Select random samples
        indices = np.random.choice(len(test_D), min(num_samples, len(test_D)), replace=False)
        
        self.eval()  # Set model to evaluation mode
        with torch.no_grad():
            for i, idx in enumerate(indices):
                # Get prediction
                input_sample = test_D[idx:idx+1].to(self.dev)
                true_sample = test_y[idx:idx+1]
                pred_sample = self.generator(input_sample) >= 0.5
                pred_prob = self.generator(input_sample).cpu()  # Raw probability predictions
                
                # Get discriminator scores
                disc_score_real = self.discriminator(input_sample, true_sample.to(self.dev)).cpu().numpy()
                disc_score_fake = self.discriminator(input_sample, pred_sample).cpu().numpy()
                
                # Format discriminator scores properly (handle any shape)
                d_score_real = disc_score_real.mean() if disc_score_real.size > 1 else disc_score_real.item()
                d_score_fake = disc_score_fake.mean() if disc_score_fake.size > 1 else disc_score_fake.item()
                
                # Move to CPU for visualization
                pred_sample = pred_sample.cpu()
                
                # Select slices for visualization (middle slice)
                depth = input_sample.shape[-1]
                # slice_idx = depth // 2  # Use middle slice for consistent visualization
                slice_idx = 3  # Use first slice for consistent visualization
                
                fig, ax = plt.subplots(2, 2, figsize=(12, 10))
                
                # Input image
                ax[0, 0].imshow(np.squeeze(test_D[idx, 0, :, :, slice_idx]).T, 'gray', vmin=0, vmax=100)
                ax[0, 0].set_title('Input Image')
                ax[0, 0].axis('off')
                
                # Ground truth
                ax[0, 1].imshow(np.squeeze(true_sample[0, 0, :, :, slice_idx]).T, 'gray')
                ax[0, 1].set_title(f'Ground Truth (D-score: {d_score_real:.3f})')
                ax[0, 1].axis('off')
                
                # Prediction (binary)
                ax[1, 0].imshow(np.squeeze(pred_sample[0, 0, :, :, slice_idx]).T, 'gray')
                ax[1, 0].set_title(f'Prediction Binary (D-score: {d_score_fake:.3f})')
                ax[1, 0].axis('off')
                
                # Prediction (probability)
                prob_img = ax[1, 1].imshow(np.squeeze(pred_prob[0, 0, :, :, slice_idx]).T, 'viridis')
                ax[1, 1].set_title('Prediction (Probability)')
                ax[1, 1].axis('off')
                
                # Add colorbar
                cbar = fig.colorbar(prob_img, ax=ax[1, 1], fraction=0.046, pad=0.04)
                cbar.set_label('Probability')
                
                # Calculate Dice score for this sample
                dice_score = self.calculate_dice_score(pred_sample, true_sample).item()
                
                # Add info about the cGAN's assessment
                gan_assessment = "Real-looking" if d_score_fake > 0.5 else "Fake-looking"
                plt.suptitle(f'Sample {idx} - Dice Score: {dice_score:.4f} - GAN assessment: {gan_assessment}', fontsize=16)
                
                plt.tight_layout()
                plt.savefig(f'{save_dir}/cgan_sample_prediction_{i}.png')
                plt.close()
                
                print(f"Sample visualization saved as {save_dir}/cgan_sample_prediction_{i}.png")
                
    def calculate_dice_loss(self, pred, target):
        """Calculate Dice loss for evaluation"""
        smooth = 1e-5
        
        # Calculate intersection and union
        intersection = torch.sum(pred * target)
        union = torch.sum(pred) + torch.sum(target)
        
        # Calculate Dice coefficient
        dice_coef = (2.0 * intersection + smooth) / (union + smooth)
        
        # Return Dice loss
        return 1.0 - dice_coef
    
    def calculate_dice_score(self, pred, target):
        """Compute Dice score with predicted and ground truth segmentations"""
        smooth = 1e-5
        intersection = 2 * torch.sum(pred * target) + smooth
        return intersection / (torch.sum(pred) + torch.sum(target) + smooth)
    
    def train_model(self, train_D, train_y, valid_D, valid_y, num_epochs=500, bs=10, lr=1e-2, 
                    weight_initial=100.0, weight_final=5.0, savebest=None, g_d_ratio=2):
        """
        Train the cGAN with all components from the project requirements
        
        Parameters:
        -----------
        train_D : tensor
            Training data input images
        train_y : tensor
            Training data ground truth segmentations
        valid_D : tensor
            Validation data input images
        valid_y : tensor
            Validation data ground truth segmentations
        num_epochs : int
            Number of epochs to train
        bs : int
            Batch size
        lr : float
            Learning rate
        weight_initial : float
            Initial positive class weight for BCE loss
        weight_final : float
            Final positive class weight for BCE loss 
        savebest : str
            Path to save best model checkpoint
        g_d_ratio : int
            Ratio of generator updates to discriminator updates (g_d_ratio:1)
        """
        # Check if a checkpoint exists and load it
        checkpoint_path = savebest if savebest else 'cGAN_chiasm.pth'
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=self.dev, weights_only=False)
            
            # Copy attributes from saved model
            self.load_state_dict(checkpoint.state_dict(), strict=False)
            self.epoch = checkpoint.epoch
            self.best_epoch = checkpoint.best_epoch
            self.tlosslist = checkpoint.tlosslist
            self.vlosslist = checkpoint.vlosslist
            
            start_epoch = self.epoch + 1
            best_val_loss = min(self.vlosslist) if self.vlosslist else float('inf')
            train_losses_G = self.tlosslist if self.tlosslist else []
            val_losses = self.vlosslist if self.vlosslist else []
            
            print(f"Resuming training from epoch {start_epoch} (checkpoint: {checkpoint_path})")
        else:
            start_epoch = 0
            best_val_loss = float('inf')
            train_losses_G = []
            val_losses = []
            
            print("Starting new training run")
        
        # Ensure validation data is on the device
        valid_D = valid_D.to(self.dev)
        valid_y = valid_y.to(self.dev)
        
        # Setup optimizers for generator and discriminator
        optimizer_G = torch.optim.SGD(self.generator.parameters(), lr=lr, momentum=0.9)
        optimizer_D = torch.optim.SGD(self.discriminator.parameters(), lr=lr*0.5, momentum=0.9)  # Lower LR for discriminator
        
        # Training records
        train_losses_D = []
        
        # Setup for plotting
        plt.figure(figsize=(10, 6))
        plt.ion()
        
        # Get the first 10 cases as labeled data
        N = train_y.size(0)
        labeled_indices = range(min(10, N))
        unlabeled_indices = range(10, N) if N > 10 else []
        
        D_labeled = train_D[labeled_indices]
        y_labeled = train_y[labeled_indices]
        
        D_unlabeled = train_D[unlabeled_indices] if unlabeled_indices else None
        
        # Begin training loop with tqdm progress bar
        epoch_pbar = trange(start_epoch, num_epochs, desc="Training Progress", position=0)
        
        for epoch in epoch_pbar:
            self.epoch = epoch  # Update the current epoch
            self.train()  # Set model to training mode
            epoch_G_losses = []
            epoch_D_losses = []
            
            # Calculate current positive class weight (warmup schedule)
            if self.epoch < 100:  # Linear decrease over first 100 epochs
                pos_weight = weight_initial - (weight_initial - weight_final) * (self.epoch / 100)
            else:
                pos_weight = weight_final
            
            pos_weight = torch.tensor([pos_weight], device=self.dev)
            
            # Shuffle data for each epoch
            labeled_shuffle = torch.randperm(len(labeled_indices))
            
            # Calculate batches
            NB = int(np.ceil(len(labeled_indices) / bs))
            
            # Process batches without the batch-level progress bar
            for i in range(NB):
                # Get batch of labeled data
                start_i = (i * bs) % len(labeled_indices)
                end_i = min(start_i + bs, len(labeled_indices))
                indices = labeled_shuffle[start_i:end_i]
                
                x_batch = D_labeled[indices].to(self.dev)
                y_batch = y_labeled[indices].to(self.dev)
                
                # Apply data augmentation
                x_aug, y_aug = self.Augment(x_batch, y_batch)
                
                # TRAINING ORDER CHANGE: Train Generator First
                # This helps prevent the discriminator from becoming too strong too quickly
                
                # Train Generator multiple times per discriminator update
                for _ in range(g_d_ratio):
                    optimizer_G.zero_grad()
                    
                    # Generate segmentation
                    fake_seg = self.generator(x_aug)
                    
                    # Component 1: BCE + Dice loss for labeled data (Loss G1)
                    # BCE loss with positive class weighting
                    loss_BCE = F.binary_cross_entropy(fake_seg, y_aug, weight=pos_weight)
                    
                    # Dice loss component
                    smooth = 1e-5
                    intersection = torch.sum(fake_seg * y_aug)
                    dice_term = (2.0 * intersection + smooth) / (torch.sum(fake_seg) + torch.sum(y_aug) + smooth)
                    loss_Dice = 1.0 - dice_term
                    
                    # Combined supervised loss
                    loss_G1 = loss_BCE + loss_Dice
                    
                    # Component 2: Adversarial loss - fool the discriminator
                    # Use adversarial loss directly
                    fake_output = self.discriminator(x_aug, fake_seg)
                    
                    # Component 3: Unlabeled data term (Loss G2)
                    loss_G2 = 0
                    if D_unlabeled is not None and len(unlabeled_indices) > 0:
                        # Get batch of unlabeled data
                        unl_indices = torch.randperm(len(unlabeled_indices))[:bs]
                        x_unlabeled = D_unlabeled[unl_indices].to(self.dev)
                        
                        # Generate segmentation for unlabeled data
                        unlabeled_seg = self.generator(x_unlabeled)
                        
                        # Get discriminator score for unlabeled data
                        disc_unlabeled_output = self.discriminator(x_unlabeled, unlabeled_seg)
                        
                        # Simple adversarial loss for unlabeled data
                        loss_G2 = F.binary_cross_entropy(disc_unlabeled_output, torch.ones_like(disc_unlabeled_output).to(self.dev) * 0.9)
                    
                    # Total generator loss with adjusted weights
                    G_loss = loss_G1 + loss_G2 + loss_Dice
                    G_loss.backward()
                    
                    # Check generator gradients
                    g_grad_stats = self.check_gradients(self.generator, name="Generator")
                    
                    # Apply gradient clipping to prevent exploding gradients
                    torch.nn.utils.clip_grad_norm_(self.generator.parameters(), 1.0)
                    optimizer_G.step()
                    
                    # Record loss
                    epoch_G_losses.append(G_loss.item())
                
                # Now train Discriminator (only once per iteration)
                optimizer_D.zero_grad()
                
                # Real samples
                real_output = self.discriminator(x_aug, y_aug)
                real_labels = torch.ones_like(real_output).to(self.dev) * 0.9  # Label smoothing
                D_real_loss = F.binary_cross_entropy(real_output, real_labels)
                
                # Fake samples - make sure to get fresh predictions
                with torch.no_grad():
                    fake_seg = self.generator(x_aug).detach()
                fake_output = self.discriminator(x_aug, fake_seg)
                fake_labels = torch.zeros_like(fake_output).to(self.dev) + 0.1  # Label smoothing
                D_fake_loss = F.binary_cross_entropy(fake_output, fake_labels)
                
                # Total discriminator loss
                D_loss = D_fake_loss
                D_loss.backward()
                
                # Get discriminator gradients after backward pass
                d_grad_stats = self.check_gradients(self.discriminator, name="Discriminator")
                
                # Apply gradient clipping to prevent exploding gradients
                torch.nn.utils.clip_grad_norm_(self.discriminator.parameters(), 1.0)
                optimizer_D.step()
                
                # Record loss
                epoch_D_losses.append(D_loss.item())

            # Validation step
            self.eval()
            with torch.no_grad():
                valid_preds = self.generator(valid_D)
                valid_loss = self.calculate_dice_loss(valid_preds, valid_y).item()
                valid_dice = self.calculate_dice_score(valid_preds, valid_y).item()
            
            # Record epoch losses
            avg_G_loss = np.mean(epoch_G_losses)
            avg_D_loss = np.mean(epoch_D_losses)
            
            train_losses_G.append(avg_G_loss)
            train_losses_D.append(avg_D_loss)
            val_losses.append(valid_loss)
            
            # Update epoch progress bar with overall status
            epoch_pbar.set_postfix({
                'G_loss': f'{avg_G_loss:.4f}',
                'Val_dice': f'{valid_dice:.4f}',
                'Best': f'{best_val_loss:.4f}',
                'D_loss': f'{avg_D_loss:.4f}',
                'Val_loss': f'{valid_loss:.4f}',
                'D_grad': f'{d_grad_stats["avg"]:.2e}',
                'G_grad': f'{g_grad_stats["avg"]:.2e}'
            })
            
            # Save best model and checkpoint current model
            if valid_loss < best_val_loss:
                best_val_loss = valid_loss
                self.best_epoch = self.epoch
                if savebest:
                    torch.save(self, savebest)
                    epoch_pbar.write(f"✓ Saved best model at epoch {self.epoch+1} with validation loss {valid_loss:.4f}")
            
            # Save checkpoint periodically (every 50 epochs)
            if (self.epoch + 1) % 50 == 0:
                checkpoint_name = f"cGAN_checkpoint_epoch_{self.epoch+1}.pth"
                torch.save(self, checkpoint_name)
                epoch_pbar.write(f"✓ Saved checkpoint at epoch {self.epoch+1}")
            
            # Update plot - Fixed the plotting to ensure x and y have the same dimensions
            plt.clf()
            epochs_completed = len(train_losses_G)
            x_range = np.arange(1, epochs_completed + 1)
            
            plt.plot(x_range, train_losses_G, 'r-', label='Generator Loss')
            plt.plot(x_range, val_losses, 'g-', label='Validation Loss')
            
            if self.best_epoch < epochs_completed:
                plt.plot(self.best_epoch+1, val_losses[self.best_epoch], 'b*', label=f'Best: {val_losses[self.best_epoch]:.4f}')
            
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.title('Training Progress')
            plt.legend()
            plt.yscale('log')
            plt.grid(True, alpha=0.3)
            # plt.pause(0.01) # Uncomment to show the plot in real-time
        
        # Save loss history
        self.tlosslist = train_losses_G
        self.vlosslist = val_losses
        
        plt.ioff()
        plt.show() # Uncomment to show the final plot
        
        return self

if __name__ == "__main__":
    # Load data
    try:
        f = open('EECE_395/chiasm.json')
        d = json.load(f)
        f.close()
    except Exception as e:
        print(f"Error loading dataset: {e}")
        exit(1)
    
    # Setup device
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {dev}")
    
    # Prepare data
    D = np.array(d['D'], dtype=np.float32)
    y = np.array(d['y'], dtype=np.float32)
    
    # Define dataset split ratios
    testf = 1/10
    validf = 1/10
    trainf = 1 - testf - validf
    
    # Random split with fixed seed for reproducibility
    np.random.seed(42)
    indices = np.random.permutation(len(y))
    
    train_size = int(trainf * len(indices))
    valid_size = int(validf * len(indices))
    
    train_indices = indices[:train_size]
    valid_indices = indices[train_size:train_size+valid_size]
    test_indices = indices[train_size+valid_size:]
    
    # Create tensors
    train_D = torch.tensor(D[train_indices])
    train_y = torch.tensor(y[train_indices, np.newaxis, :, :, :])
    valid_D = torch.tensor(D[valid_indices])
    valid_y = torch.tensor(y[valid_indices, np.newaxis, :, :, :])
    test_D = torch.tensor(D[test_indices])
    test_y = torch.tensor(y[test_indices, np.newaxis, :, :, :])

    # Calculate class imbalance to use as weights instead of a fixed 100
    weight_f = torch.sum(train_y==0)/torch.sum(train_y==1)
    print(f"Class imbalance factor: {weight_f:.2f}")

    # Create and train the model
    model = cGAN(dev)
    model = model.train_model(train_D, train_y, valid_D, valid_y, 
                             num_epochs=500, bs=5, lr=1e-2, 
                             weight_initial=weight_f, weight_final=5.0,
                             savebest='cGAN_chiasm.pth', g_d_ratio=1)
    
    # After loading or training your model:
    model.visualize_sample_predictions(test_D, test_y, num_samples=4, save_dir='sample_predictions')
    
    # Test the model on test set
    model.eval()
    with torch.no_grad():
        test_preds = model(test_D.to(dev)) >= 0.5
        dice_score = model.calculate_dice_score(test_preds, test_y.to(dev)).item()
        
    print(f"Final test Dice score: {dice_score:.4f}")