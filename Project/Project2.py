"""

Portions of this code used Claude 3.5 Sonnet as a copilot.

"""

from myVTKWin import *
import matplotlib
import numpy as np
import nrrd
from volumeViewer import *
import numpy as np
from skimage import measure
import os
import matplotlib.pyplot as plt

class surface():
    def __init__(self, img, isolevel, voxsz):

        self.verts = None
        self.faces = None
        self.img = img
        self.isolevel = None
        self.voxsz = voxsz
    
    def createSurfaceFromVolume(self):
        # Create a mesh using marching cubes
        verts, faces, normals, values = measure.marching_cubes(self.img, level=self.isolevel, spacing=self.voxsz)
        self.verts = verts
        self.faces = faces

        return verts, faces
    
    def connectedComponents(self):
        verts, faces = self.createSurfaceFromVolume()
        
        # Union-Find data structure - works much faster than heap. Thanks Claude!
        class UnionFind:
            def __init__(self, size):
                self.parent = np.arange(size)
                self.rank = np.zeros(size, dtype=np.int32)
            
            def find(self, x):
                if self.parent[x] != x:
                    self.parent[x] = self.find(self.parent[x])  # Path compression
                return self.parent[x]
            
            def union(self, x, y):
                px, py = self.find(x), self.find(y)
                if px == py:
                    return
                if self.rank[px] < self.rank[py]:
                    self.parent[px] = py
                elif self.rank[px] > self.rank[py]:
                    self.parent[py] = px
                else:
                    self.parent[py] = px
                    self.rank[px] += 1

        # Initialize Union-Find with number of vertices
        uf = UnionFind(len(verts))
        
        # Connect vertices in each face
        for face in faces:
            uf.union(face[0], face[1])
            uf.union(face[1], face[2])
        
        # Get component labels for all vertices
        labels = np.array([uf.find(i) for i in range(len(verts))])
        unique_components = np.unique(labels)
        
        # Create separate surface objects for each component
        H = []
        for component_id in unique_components:
            # Get mask for current component
            mask = (labels == component_id)
            
            # Create vertex mapping
            old_to_new = {old: new for new, old in enumerate(np.where(mask)[0])}
            
            # Get vertices for this component
            new_verts = verts[mask]
            
            # Get faces that use only vertices in this component
            valid_faces = np.all(np.isin(faces, np.where(mask)[0]), axis=1)
            component_faces = faces[valid_faces]
            
            # Remap face indices
            new_faces = np.array([[old_to_new[v] for v in face] for face in component_faces])
            
            # Create new surface object
            new_surface = surface(self.img, self.isolevel, self.voxsz)
            new_surface.verts = new_verts
            new_surface.faces = new_faces
            H.append(new_surface)
        
        return H
    
    def volume(self):
        # Volume method: compute volume (mm³)
        # Note: vertices are already scaled by voxel size from marching_cubes
        # if self.verts is None or self.faces is None:
        #     self.createSurfaceFromVolume()
            
        v1 = self.verts[self.faces[:, 0]]
        v2 = self.verts[self.faces[:, 1]]
        v3 = self.verts[self.faces[:, 2]]
        # Use signed volume formula for oriented triangles
        cross = np.cross(v2 - v1, v3 - v1)
        volume = np.abs(np.sum(np.multiply(v1, cross)) / 6.0)
        return volume

def volumeViewer_P1(config, show_surface=False, show_volume=False, connected_components=False):
    if show_surface:
        win = myVtkWin()

        for i in range(len(config.imgs_masks)):
            img, header = nrrd.read(config.imgs_masks[i])
            voxsz = [header['space directions'][0][0], header['space directions'][1][1],
                    header['space directions'][2][2]]
            
            s = surface(img, config.isolvel, voxsz)
            s.createSurfaceFromVolume()
            win.addSurf(s.verts, s.faces, color=config.cols[i][:3])

        win.start()
    
    if show_volume:
        for i in range(len(config.imgs)):
            img, header = nrrd.read(config.imgs[i])
            voxsz = [header['space directions'][0][0], 
                    header['space directions'][1][1],
                    header['space directions'][2][2]]
            
            viewer = volumeViewer()
            viewer.setImage(img, voxsz, autocontrast=True, showHistogram=True)
            
        viewer.display()
    
    if connected_components:
        win = myVtkWin()
        all_volumes_by_image = []  # List of lists, each sublist contains volumes for one image
        
        for img_idx, img_path in enumerate(config.imgs):
            # Load image
            img, header = nrrd.read(img_path)
            voxsz = [header['space directions'][0][0], header['space directions'][1][1],
                    header['space directions'][2][2]]
            
            # Add padding with -1024 (air in HU units)
            imgzp = -1024 * np.ones(np.array(img.shape) + 2)
            imgzp[1:-1, 1:-1, 1:-1] = img
            
            # Create a surface object using padded image
            s = surface(imgzp, config.isolevel, voxsz)
            
            # Get connected components
            components = s.connectedComponents()
            
            # Shift vertices back by voxel size to account for padding
            for component in components:
                if component.verts is not None:
                    component.verts[:,0] -= voxsz[0]
                    component.verts[:,1] -= voxsz[1]
                    component.verts[:,2] -= voxsz[2]
            
            volumes = []
            # Only display components for the first image
            if img_idx == 0:
                colors = matplotlib.colormaps['jet'](np.linspace(0, 1, len(components)))
                for i, component in enumerate(components):
                    if component.verts is not None and component.faces is not None:
                        color = colors[i][:3]  # Get RGB values from jet colormap
                        win.addSurf(component.verts, component.faces, color=color)
            
            # Calculate volumes for all images
            for i, component in enumerate(components):
                if component.verts is not None and component.faces is not None:
                    volume = component.volume()   
                    volumes.append(volume)
            
            all_volumes_by_image.append(volumes)

        # Create the boxplot for volumes by image
        plt.figure(figsize=(12, 6))
        plt.boxplot(all_volumes_by_image, labels=[f'Image {i+1}' for i in range(len(config.imgs))])
        plt.title('Volume Distribution by Image')
        plt.ylabel('Volume (mm³)')
        plt.xlabel('Images')
        plt.xticks(rotation=45)
        plt.show(block=False)
        plt.pause(0.1)  # Small pause to ensure plot renders

        # Print statistics
        print("\nVolume Statistics by Image:")
        for i, volumes in enumerate(all_volumes_by_image):
            print(f"Image {i+1}:")
            print(f"  Max volume: {max(volumes)}")
            print(f"  Number of components: {len(volumes)}")
        
        win.start()

class Config():
    """
    Setting up all paths and inputs
    """
    def __init__(self):
        self.isolevel = 700

        # Get the base path by finding the script location and navigating up to the root
        script_dir = os.path.dirname(os.path.commonprefix(__file__))
        self.basepath = os.path.dirname(os.path.dirname(script_dir))
        print(self.basepath)
        
        # Get the EECE_395 directory
        self.eece_path = os.path.join(self.basepath, 'EECE_395')
        
        # Get all patient folders and sort them
        self.patient_folders = sorted([f for f in os.listdir(self.eece_path) 
                                if os.path.isdir(os.path.join(self.eece_path, f))])
        self.first_n_patients = self.patient_folders[:10]  # Limit to first 10 patients

        # Get all image files (always named img.nrrd in each patient folder)
        self.imgs = []
        for patient in self.first_n_patients:
            img_path = os.path.join(self.eece_path, patient, 'img.nrrd')
            if os.path.exists(img_path):
                self.imgs.append(img_path)
        
        # Get all structure files from each patient's structures folder
        self.imgs_masks = []
        for patient in self.first_n_patients:
            structure_path = os.path.join(self.eece_path, patient, 'structures')
            if os.path.exists(structure_path):
                structures = sorted([f for f in os.listdir(structure_path) 
                                  if f.endswith('.nrrd')])
                self.imgs_masks.extend([os.path.join(structure_path, s) for s in structures])

        # Create distinct colors using matplotlib's jet colormap
        self.cols = [matplotlib.colormaps['jet'](i) for i in np.linspace(0, 1, len(self.imgs_masks))]

if __name__ == "__main__":
    # Test configuration
    config = Config()
    
    # View both surface and volume
    #volumeViewer_P1(config, show_surface=True, show_volume=False)
    volumeViewer_P1(config, connected_components=True)
