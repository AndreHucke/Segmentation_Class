# % Class to perform level set segmentation
# % gradientNB, curvatureNB, DuDt2 functions need to be added
# % ECE 8396: Medical Image Segmentation
# % Spring 2025
# % Author: Prof. Jack Noble; jack.noble@vanderbilt.edu
# % Modified by: Andre Hucke > Added testFastMarching function
# % Parts of this code were created using AI. All code was reviewed and modified by the author.

import json
from fastMarching import * # import your custom level set and fast marching solutions
from Project2 import * # import your previously defined surface class
from levelSet import * # import your level set class

def testLevelSet():
    f = open('Project/Project5.json','rt')
    d = json.load(f)
    f.close()

    crp = np.array(d['headCT'])
    voxsz = np.array(d['voxsz'])


    fig, ax = plt.subplots(1,2)
    plt.pause(0.1)
    dmapi = np.ones(np.shape(crp))
    dmapi[2:-3,2:-3,2:-3]=-1
    ls = levelSet()
    params = levelSetParams(maxiter=50, visrate=1, method='CV', reinitrate=5, mindist=7, convthrsh=1e-2, mu=2, dtt=np.linspace(3,.1,50))
    dmap = ls.segment(crp, dmapi, params, ax)

    win = myVtkWin()
    s = surface(-dmap, 0, voxsz)
    s.createSurfaceFromVolume()
    win.addSurf(s.verts, s.faces)
    win.start()

def testFastMarching():
    # Load data from the JSON file
    f = open('Project/Project5.json','rt')
    d = json.load(f)
    f.close()

    # Get test data and voxel size
    test_dmap_init = np.array(d['test_dmap_init'])
    voxsz = np.array(d['voxsz'])
    
    # Check dimensionality and fix if needed
    if len(test_dmap_init.shape) == 2:
        print(f"Original shape: {test_dmap_init.shape}")
        # Expand to 3D by adding a third dimension
        test_dmap_init = test_dmap_init[:, :, np.newaxis]
        print(f"Expanded shape: {test_dmap_init.shape}")
        
        # Adjust voxel size if needed
        if len(voxsz) == 2:
            voxsz = np.append(voxsz, 1.0)
            print(f"Adjusted voxel size: {voxsz}")

    # Initialize Fast Marching with visualization
    fm = fastMarching(plot=True)
    
    # Run the update function on the test data
    fm.update(test_dmap_init, nbdist=np.inf, voxsz=voxsz)
    
    # Calculate the mean absolute error
    result_dmap = fm.dmap
    ground_truth = test_dmap_init
    mean_abs_error = np.mean(np.abs(result_dmap - ground_truth))
    
    print(f"Mean Absolute Error: {mean_abs_error}")
    
    # Display results more simply - use the middle slice if 3D
    plt.figure(figsize=(12, 6))
    
    if len(result_dmap.shape) == 3 and result_dmap.shape[2] > 1:
        middle_slice = result_dmap.shape[2] // 2
        plt.subplot(1, 2, 1)
        plt.imshow(ground_truth[:, :, middle_slice], cmap='gray')
        plt.title('Ground Truth')
        
        plt.subplot(1, 2, 2)
        plt.imshow(result_dmap[:, :, middle_slice], cmap='gray')
        plt.title('Fast Marching Result')
    else:
        plt.subplot(1, 2, 1)
        plt.imshow(ground_truth[:, :, 0], cmap='gray')
        plt.title('Ground Truth')
        
        plt.subplot(1, 2, 2)
        plt.imshow(result_dmap[:, :, 0], cmap='gray')
        plt.title('Fast Marching Result')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Uncomment one of the lines below to run the desired test
    # testFastMarching()
    testLevelSet()