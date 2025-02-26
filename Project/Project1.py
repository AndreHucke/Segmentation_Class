from myVTKWin import *
import matplotlib
import numpy as np
import nrrd
from volumeViewer import *
import numpy as np
from skimage import measure

class surface():
    def __init__(self, img, isolevel, voxsz):

        self.verts = None
        self.faces = None
        self.img = img
        self.isolevel = isolevel
        self.voxsz = voxsz
    
    def createSurfaceFromVolume(self):
        # Create a mesh using marching cubes
        verts, faces, normals, values = measure.marching_cubes(self.img, level=self.isolevel, spacing=self.voxsz)
        self.verts = verts
        self.faces = faces
    
def Viewer_P1(config, show_surface=False, show_volume=False):
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

class Config():
    """
    Setting up all paths and inputs
    """
    isolvel = 0.5

    basepath = '/home/teixeia/projects/homeworks/Segmentation_Noble/'

    imgs_masks = []
    imgs_masks.append(basepath + 'EECE_395/0522c0001/structures/BrainStem.nrrd')
    imgs_masks.append(basepath + 'EECE_395/0522c0001/structures/OpticNerve_L.nrrd')
    imgs_masks.append(basepath + 'EECE_395/0522c0001/structures/OpticNerve_R.nrrd')
    imgs_masks.append(basepath + 'EECE_395/0522c0001/structures/Chiasm.nrrd')
    imgs_masks.append(basepath + 'EECE_395/0522c0001/structures/Mandible.nrrd')

    imgs = []
    imgs.append(basepath + 'EECE_395/0522c0001/img.nrrd')

    # Create distinct colors using matplotlib's jet colormap
    cols = [matplotlib.colormaps['jet'](i) for i in np.linspace(0, 1, len(imgs_masks))]

if __name__ == "__main__":
    # Test configuration
    config = Config()
    
    # View both surface and volume
    Viewer_P1(config, show_surface=True, show_volume=True)


