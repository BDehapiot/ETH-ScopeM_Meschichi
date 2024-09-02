#%% Imports -------------------------------------------------------------------

import time
import napari
from numba import cuda
from skimage import io
from pathlib import Path
from functions import limit_vram, segment, measure, save

#%% Comments ------------------------------------------------------------------

#%% Parameters ----------------------------------------------------------------

# Path
# data_path = Path(Path.cwd(), "data", "local") 
data_path = Path("D:/local_Meschichi/data")
# data_path = Path(r"\\scopem-idadata.ethz.ch\BDehapiot\remote_Meschichi\data")

# Batch
batch = False
overwrite = True
stack_name = "KASind2" # if batch == False 

# nMask
lmax_dist = 5
lmax_prom = 0.15
prob_thresh = 0.25
clear_nBorder = True
min_nSize = 4096

# cMask
tophat_size = 3
tophat_sigma = 1
tophat_tresh_coeff = 1.25
min_cSize = 64

# Measure
rFactor = 0.5

# GPU
vram = None # Limit vram (None to deactivate)
if vram is not None:
    limit_vram(vram)
    print(f"VRAM limited to {vram}")

#%% Execute -------------------------------------------------------------------

if __name__ == "__main__":
    
    # Get paths
    paths = []
    for path in data_path.iterdir():
        if path.suffix != ".nd2":
            continue
        
        is_dir = (path.parent / path.stem).is_dir()
        
        if batch:
           if overwrite:
               paths.append(path)
           elif not is_dir:
               paths.append(path)
        elif path.stem == stack_name:
            if overwrite:
                paths.append(path)
            elif not is_dir:
                paths.append(path)

    # Process
    for path in paths:
           
        print(f"{path.stem}")
        t0 = time.time()
                
        # Segment
        print("Segment :", end='')        
        outputs = segment(
            path,
            # nMask
            lmax_dist=lmax_dist,
            lmax_prom=lmax_prom,
            prob_thresh=prob_thresh,
            clear_nBorder=clear_nBorder,
            min_nSize=min_nSize,
            # cMask
            tophat_size=tophat_size,
            tophat_sigma=tophat_sigma,
            tophat_tresh_coeff=tophat_tresh_coeff,
            min_cSize=min_cSize,
            )
        t1 = time.time()
        print(f" {(t1-t0):<5.2f}s")
        
        # # Measure
        # print("Measure :", end='')
        # outputs = measure(outputs, rFactor=rFactor)
        # t2 = time.time()
        # print(f" {(t2-t1):<5.2f}s")
        
        # # Save
        # print("Save :", end='')
        # save(path, outputs)
        # t3 = time.time()
        # print(f" {(t3-t2):<5.2f}s")

#%%

import nd2
import numpy as np
import pandas as pd
from skimage import io
from pathlib import Path
import segmentation_models as sm
from joblib import Parallel, delayed

# Skimage
from skimage.feature import peak_local_max
from skimage.transform import rescale, resize
from skimage.measure import label, regionprops
from skimage.filters import gaussian, threshold_otsu
from skimage.segmentation import clear_border, expand_labels
from skimage.morphology import (
    disk, remove_small_objects, white_tophat
    )

# Scipy
from scipy.ndimage import distance_transform_edt

# -----------------------------------------------------------------------------

rFactor = 0.5

# -----------------------------------------------------------------------------

# Get variables
stack = outputs["stack"]
nLabels = outputs["nLabels"]
cLabels = outputs["cLabels"]
metadata = outputs["metadata"]

print("Rescale :", end='') 
t0 = time.time()

# Isotropic stacks (1st rescaling)
ratio = metadata["vZ"] / metadata["vY"]
stack = rescale(stack, (ratio * rFactor, rFactor, rFactor), order=0)
nLabels = rescale(nLabels, (ratio * rFactor, rFactor, rFactor), order=0)
cLabels = rescale(cLabels, (ratio * rFactor, rFactor, rFactor), order=0)

t1 = time.time()
print(f" {(t1-t0):<5.2f}s")

# nData -------------------------------------------------------------------

print("nData :", end='') 
t0 = time.time()

nData = {
    
    # Nuclei
    "nLabel"   : [],
    "nArea"    : [],
    "nCtrd"    : [],
    "nMajor"   : [], 
    "nMinor"   : [],
    "nMMRatio" : [],
    
    # Chromocenters
    "cLabels"  : [], 
    "cNumber"  : [],
    "cArea"    : [], 
    "cnRatio"  : [],
    
    }

for nProps in regionprops(nLabels):
    
    # Nuclei
    nData["nLabel"].append(nProps.label)
    nData["nArea"].append(nProps.area)
    nCtrd = nProps.centroid
    nCtrd = [int(ctrd) for ctrd in nCtrd]
    nData["nCtrd"].append(nCtrd)
    nData["nMajor"].append(nProps.axis_major_length)
    nData["nMinor"].append(nProps.axis_minor_length)
    nData["nMMRatio"].append(
        nProps.axis_minor_length / nProps.axis_major_length)

    # Chromocenters       
    unique = np.unique(cLabels[nLabels == nProps.label])
    unique = unique[unique != 0]
    nData["cLabels"].append(unique)
    nData["cNumber"].append(unique.shape[0])
    cMask = cLabels > 0
    cArea = np.sum(cMask[nLabels == nProps.label])
    nData["cArea"].append(cArea)
    nData["cnRatio"].append(cArea / nProps.area)

nData = pd.DataFrame(nData)

t1 = time.time()
print(f" {(t1-t0):<5.2f}s")

# cData -------------------------------------------------------------------

print("EDM :", end='') 
t0 = time.time()

# Compute EDM (2nd rescaling)
nCtrd = np.stack([ctrd for ctrd in nData["nCtrd"]])
nCtrd = ((
    (nCtrd[:, 0] * rFactor).astype(int), 
    (nCtrd[:, 1] * rFactor).astype(int), 
    (nCtrd[:, 2] * rFactor).astype(int),
    ))
EDMb = distance_transform_edt(rescale(nLabels, rFactor, order=0) > 0)
EDMc = np.zeros_like(EDMb, dtype=bool)
EDMc[nCtrd] = True
EDMc = distance_transform_edt(np.invert(EDMc))
EDMc[EDMb == 0] = 0
EDMb *= metadata["vY"] / (rFactor ** 2)
EDMc *= metadata["vY"] / (rFactor ** 2)
EDMb = resize(EDMb, nLabels.shape, order=0)
EDMc = resize(EDMc, nLabels.shape, order=0)

t1 = time.time()
print(f" {(t1-t0):<5.2f}s")

viewer = napari.Viewer()
viewer.add_image(EDMb, contrast_limits=[0, 3])
viewer.add_image(EDMc, contrast_limits=[0, 3])

          
#%%

    # t0 = time.time()
    # # Clear VRAM
    # cuda.select_device(0)
    # cuda.close()
    # t1 = time.time()
    # print(f"{t1 - t0:.5f}")
                
    # # Display 
    # scale = [outputs["metadata"]["vZ"] / outputs["metadata"]["vY"], 1, 1]
    # viewer = napari.Viewer()
    # viewer.add_image(outputs["stack"], scale=scale)
    # viewer.add_labels(outputs["nLabels"], scale=scale, opacity=0.75)
    # viewer = napari.Viewer()
    # viewer.add_image(outputs["tophat"], scale=scale)
    # viewer.add_labels(outputs["cLabels"], scale=scale, opacity=0.75)