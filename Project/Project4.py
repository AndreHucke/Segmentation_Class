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
from scipy.stats import wilcoxon  
from tqdm import tqdm

class MandibleAnalysis:
    """Class to analyze and compare multiple segmentations of the mandible."""
    
    def __init__(self, base_path=None):
        """
        Initialize with base path to the dataset.
        
        Args:
            base_path: Path to the base directory of the dataset
        """
        if base_path is None:
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

    def point_to_triangle_distance(self, point, triangle_vertices):
        """
        Calculate the minimum distance from a point to a triangle.
        
        Args:
            point: A 3D point [x, y, z]
            triangle_vertices: Array of 3 vertices defining the triangle [v1, v2, v3]
            
        Returns:
            distance: The minimum distance from point to triangle
            closest_point: The closest point on the triangle
        """
        q1, q2, q3 = triangle_vertices
        p = point
        
        v1 = q2 - q1
        v2 = q3 - q1
        
        V = np.column_stack((v1, v2))
        
        try:
            pinv = np.linalg.pinv(V)
            ab = pinv @ (p - q1)[:, np.newaxis]
            a, b = ab.flatten()
            
            if 0 <= a <= 1 and 0 <= b <= 1 and a + b <= 1:
                closest = q1 + a * v1 + b * v2
                return np.linalg.norm(p - closest), closest
            
            v3 = q3 - q2
            
            c = np.dot(v1, (p - q1)) / np.dot(v1, v1)
            d = np.dot(v2, (p - q1)) / np.dot(v2, v2)
            e = np.dot(v3, (p - q2)) / np.dot(v3, v3)
            
            c = max(0, min(1, c))
            d = max(0, min(1, d))
            e = max(0, min(1, e))
            
            dist_edge1 = np.linalg.norm(q1 + c * v1 - p)
            dist_edge2 = np.linalg.norm(q1 + d * v2 - p)
            dist_edge3 = np.linalg.norm(q2 + e * v3 - p)
            
            min_dist = min(dist_edge1, dist_edge2, dist_edge3)
            
            if min_dist == dist_edge1:
                closest = q1 + c * v1
            elif min_dist == dist_edge2:
                closest = q1 + d * v2
            else:
                closest = q2 + e * v3
                
            return min_dist, closest
            
        except np.linalg.LinAlgError:
            dist_v1 = np.linalg.norm(p - q1)
            dist_v2 = np.linalg.norm(p - q2)
            dist_v3 = np.linalg.norm(p - q3)
            
            min_dist = min(dist_v1, dist_v2, dist_v3)
            
            if min_dist == dist_v1:
                closest = q1
            elif min_dist == dist_v2:
                closest = q2
            else:
                closest = q3
                
            return min_dist, closest
    
    def surface_distances(self, surface1, surface2, max_dist=float('inf')):
        """
        Calculate surface distances between two surfaces.
        
        Args:
            surface1: Tuple of (vertices, faces) for the first surface
            surface2: Tuple of (vertices, faces) for the second surface
            max_dist: Maximum distance to consider (for efficiency)
            
        Returns:
            distances: Array of distances from surface1 to surface2
            mean_dist: Mean Average Surface Distance
            hausdorff_dist: Hausdorff Distance
            matching_point1: Point on surface1 with maximum distance
            matching_point2: Corresponding closest point on surface2
        """
        verts1, faces1 = surface1
        verts2, faces2 = surface2
        
        if verts1 is None or verts2 is None or faces1 is None or faces2 is None:
            return [], 0, 0, [], []
            
        distances = []
        max_distance = 0
        mp1 = np.zeros(3)
        mp2 = np.zeros(3)
        
        for vertex in tqdm(verts1, desc="Calculating surface distances", disable=len(verts1) < 100):
            min_dist = max_dist
            closest_point = None
            
            for face_idx in range(len(faces2)):
                face = faces2[face_idx]
                triangle_verts = [verts2[face[0]], verts2[face[1]], verts2[face[2]]]
                
                dist, closest = self.point_to_triangle_distance(vertex, triangle_verts)
                
                if dist < min_dist:
                    min_dist = dist
                    closest_point = closest
            
            distances.append(min_dist)
            
            if min_dist > max_distance:
                max_distance = min_dist
                mp1 = vertex
                mp2 = closest_point
        
        mean_dist = np.mean(distances) if distances else 0
        
        return distances, mean_dist, max_distance, mp1, mp2
    
    def calculate_surface_metrics(self, surface1, surface2):
        """
        Calculate bidirectional surface metrics between two surfaces.
        
        Args:
            surface1: Tuple of (vertices, faces) for the first surface
            surface2: Tuple of (vertices, faces) for the second surface
            
        Returns:
            dictionary of metrics including MASD and Hausdorff distance
        """
        if surface1[0] is None or surface2[0] is None:
            return {
                'avg_symm_surface_dist': 0,
                'hausdorff_dist': 0
            }
        
        _, mean_dist_1to2, max_dist_1to2, _, _ = self.surface_distances(surface1, surface2)
        _, mean_dist_2to1, max_dist_2to1, _, _ = self.surface_distances(surface2, surface1)
        
        avg_symm_dist = (mean_dist_1to2 + mean_dist_2to1) / 2
        hausdorff_dist = max(max_dist_1to2, max_dist_2to1)
        
        return {
            'avg_symm_surface_dist': avg_symm_dist,
            'hausdorff_dist': hausdorff_dist
        }

    def _analyze_single_case(self, case, show_visualization=True):
        """Analyze a single case."""
        print(f"\nAnalyzing case: {case}")
        
        gt_mask, gt_voxel_size, gt_header = self.load_mask(case, 'Mandible.nrrd')
        
        rater_masks = {}
        for i in range(1, 4):
            mask, voxel_size, _ = self.load_mask(case, f'target{i}.nrrd')
            rater_masks[f'rater{i}'] = mask
        
        if gt_mask is None or any(mask is None for mask in rater_masks.values()):
            print(f"Skipping case {case} due to missing masks")
            return None
        
        local_confusion_matrices = {}
        for rater, mask in rater_masks.items():
            local_confusion_matrices[rater] = self.compute_confusion_matrix(gt_mask, mask)
        
        majority_mask = self.create_majority_vote(
            rater_masks['rater1'], 
            rater_masks['rater2'], 
            rater_masks['rater3']
        )
        
        surfaces = {}
        surfaces['ground_truth'] = self.create_isosurface(gt_mask, gt_voxel_size)
        
        for rater, mask in rater_masks.items():
            surfaces[rater] = self.create_isosurface(mask, gt_voxel_size)
            
        surfaces['majority'] = self.create_isosurface(majority_mask, gt_voxel_size)
        
        volumes = {}
        volumes['ground_truth'] = self.calculate_volume(*surfaces['ground_truth'])
        
        for rater in rater_masks.keys():
            volumes[rater] = self.calculate_volume(*surfaces[rater])
            
        volumes['majority'] = self.calculate_volume(*surfaces['majority'])
        
        metrics = {}
        for rater in list(rater_masks.keys()) + ['majority']:
            dice = self.dice_coefficient(gt_mask, rater_masks[rater] if rater != 'majority' else majority_mask)
            
            surface_metrics = self.calculate_surface_metrics(surfaces['ground_truth'], surfaces[rater])
            
            metrics[rater] = {
                'dice': dice,
                'avg_symm_surface_dist': surface_metrics['avg_symm_surface_dist'],
                'hausdorff_dist': surface_metrics['hausdorff_dist']
            }
            
            print(f"{rater.capitalize()} - Volume: {volumes[rater]:.2f} mm³, " +
                  f"Dice: {dice:.4f}, " +
                  f"ASSD: {surface_metrics['avg_symm_surface_dist']:.4f} mm, " +
                  f"HD: {surface_metrics['hausdorff_dist']:.4f} mm")
        
        case_results = {
            'case': case,
            'ground_truth_volume': volumes['ground_truth']
        }
        
        for rater in list(rater_masks.keys()) + ['majority']:
            case_results[f'{rater}_volume'] = volumes[rater]
            case_results[f'{rater}_dice'] = metrics[rater]['dice']
            case_results[f'{rater}_assd'] = metrics[rater]['avg_symm_surface_dist']
            case_results[f'{rater}_hd'] = metrics[rater]['hausdorff_dist']
            
        for rater in rater_masks.keys():
            sensitivity, specificity = self.compute_sensitivity_specificity(local_confusion_matrices[rater])
            case_results[f'{rater}_sensitivity'] = sensitivity
            case_results[f'{rater}_specificity'] = specificity
        
        if show_visualization:
            vis_data = {
                'surfaces': surfaces,
                'colors': self.colors
            }
            return {
                'results': case_results,
                'confusion_matrices': local_confusion_matrices,
                'visualization': vis_data
            }
        
        return {
            'results': case_results,
            'confusion_matrices': local_confusion_matrices
        }

    def perform_wilcoxon_tests(self, results_df):
        """Perform Wilcoxon signed-rank tests between raters for each metric."""
        print("\n=== Wilcoxon Signed-Rank Test Results ===")
        
        metrics = ['dice', 'assd', 'hd']
        metric_names = {
            'dice': 'Dice Coefficient',
            'assd': 'Average Symmetric Surface Distance',
            'hd': 'Hausdorff Distance'
        }
        
        raters = ['rater1', 'rater2', 'rater3', 'majority']
        
        for metric in metrics:
            print(f"\n{metric_names[metric]}:")
            
            p_values = np.zeros((len(raters), len(raters)))
            
            for i, rater1 in enumerate(raters):
                for j, rater2 in enumerate(raters):
                    if i >= j:
                        continue
                    
                    col1 = f'{rater1}_{metric}'
                    col2 = f'{rater2}_{metric}'
                    if col1 in results_df.columns and col2 in results_df.columns:
                        _, p_value = wilcoxon(results_df[col1], results_df[col2])
                        p_values[i, j] = p_value
                        p_values[j, i] = p_value
            
            p_df = pd.DataFrame(p_values, index=raters, columns=raters)
            
            p_df_stacked = p_df.stack()
            p_df_stacked = p_df_stacked.map(lambda x: f"{x:.4f}{'*' if x < 0.05 else ''}" if x > 0 else "-")
            p_df = p_df_stacked.unstack()
            
            p_df.index = [r.capitalize() for r in raters]
            p_df.columns = [r.capitalize() for r in raters]
            
            print(p_df)
            print("* indicates statistically significant difference (p < 0.05)")
        
        return

    def create_boxplots(self, results_df):
        """Create boxplots for the different metrics."""
        fig, axs = plt.subplots(2, 2, figsize=(14, 10))
        
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
        
        if 'rater1_assd' in results_df.columns:
            assd_data = [
                results_df['rater1_assd'],
                results_df['rater2_assd'],
                results_df['rater3_assd'],
                results_df['majority_assd']
            ]
            
            axs[1, 0].boxplot(assd_data)
            axs[1, 0].set_title('Average Symmetric Surface Distance')
            axs[1, 0].set_xticklabels(['Rater 1', 'Rater 2', 'Rater 3', 'Majority'])
            axs[1, 0].set_ylabel('Distance (mm)')
            
        if 'rater1_hd' in results_df.columns:
            hd_data = [
                results_df['rater1_hd'],
                results_df['rater2_hd'],
                results_df['rater3_hd'],
                results_df['majority_hd']
            ]
            
            axs[1, 1].boxplot(hd_data)
            axs[1, 1].set_title('Hausdorff Distance')
            axs[1, 1].set_xticklabels(['Rater 1', 'Rater 2', 'Rater 3', 'Majority'])
            axs[1, 1].set_ylabel('Distance (mm)')
        
        plt.tight_layout()
        plt.pause(0.1)

    def visualize_confusion_matrices(self):
        """Create visualizations of the confusion matrices."""
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        for i, (rater, cm) in enumerate(self.confusion_matrices.items()):
            cm_norm = cm.astype('float') / cm.sum()
            
            sns.heatmap(cm_norm, annot=cm, fmt="d", cmap="Blues", 
                        xticklabels=['Negative', 'Positive'],
                        yticklabels=['Negative', 'Positive'],
                        ax=axes[i], cbar=False)
            
            sensitivity, specificity = self.compute_sensitivity_specificity(cm)
            
            axes[i].set_title(f"{rater.capitalize()} Confusion Matrix\nSensitivity: {sensitivity:.4f}, Specificity: {specificity:.4f}")
            axes[i].set_ylabel('Ground Truth')
            axes[i].set_xlabel('Rater Prediction')
        
        plt.tight_layout()
        plt.show(block=False)
        plt.pause(0.1)

    def run_analysis(self, show_visualization=True):
        """Run analysis for all cases and aggregate results."""
        self.results = []
        for rater in self.confusion_matrices:
            self.confusion_matrices[rater] = np.zeros((2, 2), dtype=np.int64)
        
        all_windows = []
        
        for case in tqdm(self.cases, desc="Processing cases"):
            result = self._analyze_single_case(case, show_visualization)
            
            if result:
                for rater, cm in result['confusion_matrices'].items():
                    self.confusion_matrices[rater] += cm
                
                self.results.append(result['results'])
                
                if show_visualization and result['visualization']:
                    win = myVtkWin(title=f"Mandible Segmentation - Case {case}")
                    vis_data = result['visualization']
                    
                    for surface_type, (verts, faces) in vis_data['surfaces'].items():
                        win.addSurf(verts, faces, 
                                  color=self.colors[surface_type], 
                                  opacity=0.3)
                    
                    win.render()
                    all_windows.append(win)
        
        df = pd.DataFrame(self.results)
        
        self.print_confusion_matrices_table()
        self.perform_wilcoxon_tests(df)
        self.create_boxplots(df)
        self.visualize_confusion_matrices()
        
        if show_visualization and all_windows:
            all_windows[0].start()
        
        return df

if __name__ == "__main__":
    analysis = MandibleAnalysis()
    results = analysis.run_analysis(show_visualization=False)
    
    input("Press Enter to close all windows and exit...")