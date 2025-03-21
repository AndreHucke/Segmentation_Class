"""
Project 4: Similarity Metrics
Name: Andre Hucke
Course: ECE 8396-02 Special Topics - Medical Image Segmentation
Professor: Dr. Noble

Parts of this code were created with the assistance of Claude 3.7. All code was reviewed and modified by the author to ensure correctness and clarity.
"""

import os
import numpy as np
import nrrd
import matplotlib.pyplot as plt
from myVTKWin import *
from skimage import measure
import pandas as pd
import seaborn as sns
from scipy.stats import wilcoxon  # Add import for Wilcoxon test

class MandibleAnalysis:
    """Class to analyze and compare multiple segmentations of the mandible."""
    
    def __init__(self, base_path=None):
        """Initialize with base path to the dataset."""
        if base_path is None:
            # Get the base path by finding the script location and navigating up to the root
            script_dir = os.path.dirname(os.path.abspath(__file__))
            self.base_path = os.path.dirname(os.path.dirname(script_dir))
        else:
            self.base_path = base_path
        
        # Path to the EECE_395 directory
        self.eece_path = os.path.join(self.base_path, 'Segmentation_Noble/EECE_395')
        
        # Get all patient folders and sort them alphabetically
        self.patient_folders = sorted([f for f in os.listdir(self.eece_path) 
                                if os.path.isdir(os.path.join(self.eece_path, f))])
        # Take only the first 10 cases
        self.cases = self.patient_folders[:10]
        
        # Define colors for visualization
        self.colors = {
            'ground_truth': [0.0, 1.0, 0.0],  # Green
            'rater1': [1.0, 0.0, 0.0],        # Red
            'rater2': [0.0, 0.0, 1.0],        # Blue
            'rater3': [1.0, 1.0, 0.0],        # Yellow
            'majority': [1.0, 0.0, 1.0]       # Magenta
        }
        
        # Results storage
        self.results = []

        # Additional storage for confusion matrix data
        self.confusion_matrices = {
            'rater1': np.zeros((2, 2), dtype=np.int64),
            'rater2': np.zeros((2, 2), dtype=np.int64),
            'rater3': np.zeros((2, 2), dtype=np.int64)
        }

    def load_mask(self, case, mask_name):
        """Load a mask file for a specific case."""
        file_path = os.path.join(self.eece_path, case, 'structures', mask_name)
        if os.path.exists(file_path):
            data, header = nrrd.read(file_path)
            voxel_size = [
                header['space directions'][0][0],
                header['space directions'][1][1],
                header['space directions'][2][2]
            ]
            return data, voxel_size, header
        else:
            print(f"Mask {mask_name} not found for case {case}")
            return None, None, None

    def create_isosurface(self, data, voxel_size, isolevel=0.5):
        """Create an isosurface from a mask using marching cubes."""
        if data is None:
            return None, None
        
        verts, faces, _, _ = measure.marching_cubes(data, level=isolevel, spacing=voxel_size)
        return verts, faces

    def calculate_volume(self, verts, faces):
        """Calculate the volume of a surface mesh."""
        if verts is None or faces is None:
            return 0
        
        # Get vertices for each triangle
        v1 = verts[faces[:, 0]]
        v2 = verts[faces[:, 1]]
        v3 = verts[faces[:, 2]]
        
        # Calculate signed volume using cross product method
        cross = np.cross(v2 - v1, v3 - v1)
        volume = np.abs(np.sum(np.multiply(v1, cross)) / 6.0)
        return volume

    def dice_coefficient(self, mask1, mask2):
        """Calculate Dice similarity coefficient between two binary masks."""
        if mask1 is None or mask2 is None:
            return 0
        
        intersection = np.sum(mask1 * mask2)
        return 2.0 * intersection / (np.sum(mask1) + np.sum(mask2))

    def hausdorff_distance(self, verts1, verts2):
        """Calculate Hausdorff distance between two point sets."""
        if verts1 is None or verts2 is None or len(verts1) == 0 or len(verts2) == 0:
            return float('inf')
        
        # Calculate distances from verts1 to verts2
        distances_1_to_2 = np.min(np.sqrt(np.sum((verts1[:, np.newaxis, :] - verts2[np.newaxis, :, :]) ** 2, axis=2)), axis=1)
        
        # Calculate distances from verts2 to verts1
        distances_2_to_1 = np.min(np.sqrt(np.sum((verts2[:, np.newaxis, :] - verts1[np.newaxis, :, :]) ** 2, axis=2)), axis=1)
        
        # Hausdorff distance is the maximum of the two directed distances
        return max(np.max(distances_1_to_2), np.max(distances_2_to_1))

    def mean_surface_distance(self, verts1, verts2):
        """Calculate mean symmetric surface distance between two point sets."""
        if verts1 is None or verts2 is None or len(verts1) == 0 or len(verts2) == 0:
            return float('inf')
        
        # Calculate distances from verts1 to verts2
        distances_1_to_2 = np.min(np.sqrt(np.sum((verts1[:, np.newaxis, :] - verts2[np.newaxis, :, :]) ** 2, axis=2)), axis=1)
        
        # Calculate distances from verts2 to verts1
        distances_2_to_1 = np.min(np.sqrt(np.sum((verts2[:, np.newaxis, :] - verts1[np.newaxis, :, :]) ** 2, axis=2)), axis=1)
        
        # Mean symmetric surface distance
        return (np.mean(distances_1_to_2) + np.mean(distances_2_to_1)) / 2.0

    def create_majority_vote(self, mask1, mask2, mask3):
        """Create a majority vote mask from three rater masks."""
        if mask1 is None or mask2 is None or mask3 is None:
            return None
        
        # Sum the masks and apply threshold (≥ 2 raters)
        sum_mask = mask1 + mask2 + mask3
        majority_mask = (sum_mask >= 2).astype(np.float32)
        return majority_mask

    def compute_confusion_matrix(self, ground_truth, rater_mask):
        """
        Compute confusion matrix between ground truth and rater mask.
        Returns a 2x2 matrix: [[TN, FP], [FN, TP]]
        """
        if ground_truth is None or rater_mask is None:
            return np.zeros((2, 2), dtype=np.int64)
        
        # Convert to binary
        gt_binary = ground_truth > 0.5
        rater_binary = rater_mask > 0.5
        
        # True Negatives (both 0)
        tn = np.sum((~gt_binary) & (~rater_binary))
        
        # False Positives (gt=0, rater=1)
        fp = np.sum((~gt_binary) & rater_binary)
        
        # False Negatives (gt=1, rater=0)
        fn = np.sum(gt_binary & (~rater_binary))
        
        # True Positives (both 1)
        tp = np.sum(gt_binary & rater_binary)
        
        return np.array([[tn, fp], [fn, tp]], dtype=np.int64)

    def compute_sensitivity_specificity(self, confusion_matrix):
        """
        Compute sensitivity (recall) and specificity from confusion matrix.
        """
        tn, fp = confusion_matrix[0]
        fn, tp = confusion_matrix[1]
        
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        
        return sensitivity, specificity

    def analyze_case(self, case, show_visualization=True):
        """Analyze a single case: load masks, calculate metrics, and visualize."""
        print(f"\nAnalyzing case: {case}")
        
        # Load the ground truth mask (mandible.nrrd)
        gt_mask, gt_voxel_size, gt_header = self.load_mask(case, 'Mandible.nrrd')
        
        # Load rater masks (target1.nrrd, target2.nrrd, target3.nrrd)
        rater_masks = {}
        for i in range(1, 4):
            mask, voxel_size, _ = self.load_mask(case, f'target{i}.nrrd')
            rater_masks[f'rater{i}'] = mask
        
        # Skip this case if any mask is missing
        if gt_mask is None or any(mask is None for mask in rater_masks.values()):
            print(f"Skipping case {case} due to missing masks")
            return
        
        # Compute confusion matrices for each rater
        confusion_matrices = {}
        for rater, mask in rater_masks.items():
            cm = self.compute_confusion_matrix(gt_mask, mask)
            confusion_matrices[rater] = cm
            # Accumulate to global confusion matrices
            self.confusion_matrices[rater] += cm
        
        # Create majority vote segmentation
        majority_mask = self.create_majority_vote(
            rater_masks['rater1'], 
            rater_masks['rater2'], 
            rater_masks['rater3']
        )
        
        # Create isosurfaces
        surfaces = {}
        surfaces['ground_truth'] = self.create_isosurface(gt_mask, gt_voxel_size)
        
        for rater, mask in rater_masks.items():
            surfaces[rater] = self.create_isosurface(mask, gt_voxel_size)
            
        surfaces['majority'] = self.create_isosurface(majority_mask, gt_voxel_size)
        
        # Calculate volumes
        volumes = {}
        volumes['ground_truth'] = self.calculate_volume(*surfaces['ground_truth'])
        
        for rater in rater_masks.keys():
            volumes[rater] = self.calculate_volume(*surfaces[rater])
            
        volumes['majority'] = self.calculate_volume(*surfaces['majority'])
        
        # Calculate metrics for each rater compared to ground truth
        metrics = {}
        for rater in list(rater_masks.keys()) + ['majority']:
            dice = self.dice_coefficient(gt_mask, rater_masks[rater] if rater != 'majority' else majority_mask)
            hausdorff = self.hausdorff_distance(surfaces['ground_truth'][0], surfaces[rater][0])
            mean_dist = self.mean_surface_distance(surfaces['ground_truth'][0], surfaces[rater][0])
            
            metrics[rater] = {
                'dice': dice,
                'hausdorff': hausdorff,
                'mean_distance': mean_dist
            }
            
            # Print results
            print(f"{rater.capitalize()} - Volume: {volumes[rater]:.2f} mm³, " +
                  f"Dice: {dice:.4f}, " +
                  f"Hausdorff: {hausdorff:.4f} mm, " +
                  f"Mean Surface Distance: {mean_dist:.4f} mm")
        
        # Store results for later aggregation
        case_results = {
            'case': case,
            'ground_truth_volume': volumes['ground_truth']
        }
        
        for rater in list(rater_masks.keys()) + ['majority']:
            case_results[f'{rater}_volume'] = volumes[rater]
            case_results[f'{rater}_dice'] = metrics[rater]['dice']
            case_results[f'{rater}_hausdorff'] = metrics[rater]['hausdorff']
            case_results[f'{rater}_mean_distance'] = metrics[rater]['mean_distance']
            
        # Add confusion matrix metrics to case results
        for rater in rater_masks.keys():
            sensitivity, specificity = self.compute_sensitivity_specificity(confusion_matrices[rater])
            case_results[f'{rater}_sensitivity'] = sensitivity
            case_results[f'{rater}_specificity'] = specificity

        self.results.append(case_results)
        
        # Visualization
        if show_visualization:
            win = myVtkWin(title=f"Mandible Segmentation - Case {case}")
            
            # Add ground truth surface
            win.addSurf(*surfaces['ground_truth'], color=self.colors['ground_truth'], opacity=0.3)
            
            # Add rater surfaces
            for rater in rater_masks.keys():
                win.addSurf(*surfaces[rater], color=self.colors[rater], opacity=0.3)
            
            # Add majority vote surface
            win.addSurf(*surfaces['majority'], color=self.colors['majority'], opacity=0.3)
            
            win.render()
            return win
        
        return None

    def print_confusion_matrices_table(self):
        """Create tabular display of confusion matrices and sensitivity/specificity values."""
        print("\n=== Confusion Matrices and Performance Metrics ===")
        
        # Create dataframe for confusion matrix results
        cm_data = []
        for rater, cm in self.confusion_matrices.items():
            sensitivity, specificity = self.compute_sensitivity_specificity(cm)
            tn, fp = cm[0]
            fn, tp = cm[1]
            accuracy = (tp + tn) / (tp + tn + fp + fn)
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            
            cm_data.append({
                'Rater': rater.capitalize(),
                'TP': tp,
                'TN': tn, 
                'FP': fp,
                'FN': fn,
                'Total Voxels': tp + tn + fp + fn,
                'Sensitivity': sensitivity,
                'Specificity': specificity,
                'Accuracy': accuracy,
                'Precision': precision
            })
        
        # Create and print the dataframe
        cm_df = pd.DataFrame(cm_data)
        cm_df = cm_df.set_index('Rater')
        
        # Print the confusion matrix components first
        print("\nConfusion Matrix Components:")
        print(cm_df[['TP', 'TN', 'FP', 'FN', 'Total Voxels']])
        
        # Print the performance metrics separately
        print("\nPerformance Metrics:")
        metrics_df = cm_df[['Sensitivity', 'Specificity', 'Accuracy', 'Precision']]
        
        # Format floating point numbers for better display
        for col in metrics_df.columns:
            metrics_df[col] = metrics_df[col].map('{:.4f}'.format)
            
        print(metrics_df)
        
        return cm_df

    def perform_wilcoxon_tests(self, results_df):
        """Perform Wilcoxon signed-rank tests between raters for each metric."""
        print("\n=== Wilcoxon Signed-Rank Test Results ===")
        
        # Metrics to compare
        metrics = ['dice', 'hausdorff', 'mean_distance']
        metric_names = {
            'dice': 'Dice Coefficient',
            'hausdorff': 'Hausdorff Distance',
            'mean_distance': 'Mean Surface Distance'
        }
        
        # Raters to compare (including majority vote)
        raters = ['rater1', 'rater2', 'rater3', 'majority']
        
        # For each metric, perform pairwise comparisons
        for metric in metrics:
            print(f"\n{metric_names[metric]}:")
            
            # Create matrix to store p-values
            p_values = np.zeros((len(raters), len(raters)))
            
            # Perform pairwise Wilcoxon tests
            for i, rater1 in enumerate(raters):
                for j, rater2 in enumerate(raters):
                    if i >= j:  # Skip diagonal and lower triangle (redundant)
                        continue
                    
                    # Perform Wilcoxon test
                    col1 = f'{rater1}_{metric}'
                    col2 = f'{rater2}_{metric}'
                    _, p_value = wilcoxon(results_df[col1], results_df[col2])
                    p_values[i, j] = p_value
                    p_values[j, i] = p_value  # Mirror for symmetric display
            
            # Create dataframe for prettier display
            p_df = pd.DataFrame(p_values, index=raters, columns=raters)
            
            # Format the dataframe using stack/map/unstack instead of deprecated applymap
            p_df_stacked = p_df.stack()
            p_df_stacked = p_df_stacked.map(lambda x: f"{x:.4f}{'*' if x < 0.05 else ''}" if x > 0 else "-")
            p_df = p_df_stacked.unstack()
            
            # Set display names
            p_df.index = [r.capitalize() for r in raters]
            p_df.columns = [r.capitalize() for r in raters]
            
            print(p_df)
            print("* indicates statistically significant difference (p < 0.05)")
        
        return
    
    def create_boxplots(self, results_df):
        """Create boxplots for the different metrics."""
        fig, axs = plt.subplots(2, 2, figsize=(14, 10))
        
        # Volume comparison
        volume_data = [
            results_df['ground_truth_volume'],
            results_df['rater1_volume'],
            results_df['rater2_volume'],
            results_df['rater3_volume'],
            results_df['majority_volume']
        ]
        
        axs[0, 0].boxplot(volume_data)
        axs[0, 0].set_title('Segmentation Volumes')
        axs[0, 0].set_xticklabels(['Ground Truth', 'Rater 1', 'Rater 2', 'Rater 3', 'Majority'])
        axs[0, 0].set_ylabel('Volume (mm³)')
        
        # Dice coefficient
        dice_data = [
            results_df['rater1_dice'],
            results_df['rater2_dice'],
            results_df['rater3_dice'],
            results_df['majority_dice']
        ]
        
        axs[0, 1].boxplot(dice_data)
        axs[0, 1].set_title('Dice Similarity Coefficient')
        axs[0, 1].set_xticklabels(['Rater 1', 'Rater 2', 'Rater 3', 'Majority'])
        axs[0, 1].set_ylabel('Dice Coefficient')
        
        # Hausdorff distance
        hausdorff_data = [
            results_df['rater1_hausdorff'],
            results_df['rater2_hausdorff'],
            results_df['rater3_hausdorff'],
            results_df['majority_hausdorff']
        ]
        
        axs[1, 0].boxplot(hausdorff_data)
        axs[1, 0].set_title('Hausdorff Distance')
        axs[1, 0].set_xticklabels(['Rater 1', 'Rater 2', 'Rater 3', 'Majority'])
        axs[1, 0].set_ylabel('Distance (mm)')
        
        # Mean surface distance
        mean_dist_data = [
            results_df['rater1_mean_distance'],
            results_df['rater2_mean_distance'],
            results_df['rater3_mean_distance'],
            results_df['majority_mean_distance']
        ]
        
        axs[1, 1].boxplot(mean_dist_data)
        axs[1, 1].set_title('Mean Surface Distance')
        axs[1, 1].set_xticklabels(['Rater 1', 'Rater 2', 'Rater 3', 'Majority'])
        axs[1, 1].set_ylabel('Distance (mm)')
        
        plt.tight_layout()
        plt.pause(0.1)  # Small pause to ensure plot renders

    def visualize_confusion_matrices(self):
        """Create visualizations of the confusion matrices."""
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        for i, (rater, cm) in enumerate(self.confusion_matrices.items()):
            # Create normalized confusion matrix for better visualization
            cm_norm = cm.astype('float') / cm.sum()
            
            # Create heatmap
            sns.heatmap(cm_norm, annot=cm, fmt="d", cmap="Blues", 
                        xticklabels=['Negative', 'Positive'],
                        yticklabels=['Negative', 'Positive'],
                        ax=axes[i], cbar=False)
            
            # Calculate and display sensitivity and specificity
            sensitivity, specificity = self.compute_sensitivity_specificity(cm)
            
            axes[i].set_title(f"{rater.capitalize()} Confusion Matrix\nSensitivity: {sensitivity:.4f}, Specificity: {specificity:.4f}")
            axes[i].set_ylabel('Ground Truth')
            axes[i].set_xlabel('Rater Prediction')
        
        plt.tight_layout()
        plt.show(block=False)
        plt.pause(0.1)  # Small pause to ensure plot renders

    def run_analysis(self, show_visualization=True):
        """Run analysis for all cases and aggregate results."""
        # Reset confusion matrices before running analysis
        for rater in self.confusion_matrices:
            self.confusion_matrices[rater] = np.zeros((2, 2), dtype=np.int64)
        
        all_windows = []
        
        for case in self.cases:
            win = self.analyze_case(case, show_visualization)
            if win:
                all_windows.append(win)
        
        # Create summary dataframe
        df = pd.DataFrame(self.results)
        
        # Calculate and print summary statistics
        print("\n=== Summary Statistics ===")
        for metric in ['volume', 'dice', 'hausdorff', 'mean_distance']:
            for rater in ['rater1', 'rater2', 'rater3', 'majority']:
                col = f'{rater}_{metric}'
                if col in df.columns:
                    mean_val = df[col].mean()
                    std_val = df[col].std()
                    print(f"{rater.capitalize()} {metric}: Mean={mean_val:.4f}, StdDev={std_val:.4f}")
        
        # Print confusion matrices
        print("\n=== Confusion Matrices ===")
        for rater, cm in self.confusion_matrices.items():
            print(f"\n{rater.capitalize()}:")
            print(f"[[TN={cm[0,0]}, FP={cm[0,1]}],")
            print(f" [FN={cm[1,0]}, TP={cm[1,1]}]]")
            
            # Calculate and print sensitivity/specificity for the aggregated data
            sensitivity, specificity = self.compute_sensitivity_specificity(cm)
            print(f"Sensitivity: {sensitivity:.4f}")
            print(f"Specificity: {specificity:.4f}")

        # Print tabular results for confusion matrices
        self.print_confusion_matrices_table()
        
        # Perform and display Wilcoxon test results
        self.perform_wilcoxon_tests(df)
        
        # Create box plots for the metrics
        self.create_boxplots(df)
        
        # Create confusion matrix visualizations
        self.visualize_confusion_matrices()
        
        # Display the first window interactively if visualization is enabled
        if show_visualization and all_windows:
            all_windows[0].start()
        return df

if __name__ == "__main__":
    analysis = MandibleAnalysis()
    results = analysis.run_analysis(show_visualization=False)
    
    # Wait for user input to keep windows open
    input("Press Enter to close all windows and exit...")