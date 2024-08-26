#%% Imports -------------------------------------------------------------------

import time
import napari
import numpy as np
from skimage import io
from pathlib import Path
from joblib import Parallel, delayed
from functions import  limit_vram, open_stack, format_stack, predict

# Skimage
from skimage.measure import label
from skimage.transform import rescale, resize
from skimage.segmentation import clear_border
from skimage.filters import gaussian, threshold_otsu
from skimage.morphology import (
    disk, remove_small_objects, binary_dilation, white_tophat
    )

#%% Parameters ----------------------------------------------------------------

# Path
# data_path = Path(Path.cwd(), "data", "local") 
data_path = Path("D:/local_Meschichi/data")
# data_path = Path(r"\\scopem-idadata.ethz.ch\BDehapiot\remote_Meschichi\data")

# GPU
vram = None # Limit vram (None to deactivate)
if vram is not None:
    limit_vram(vram)
    print(f"VRAM limited to {vram}")

# nMask
prob_sigma = 2
prob_thresh = 0.5
clear_nBorder = True
min_nSize = 4096

# cMask
tophat_size = 3
tophat_sigma = 1
tophat_tresh_coeff = 1.25
min_cSize = 32
              
#%% Functions -----------------------------------------------------------------

def process(
        path,
        prob_sigma=2,
        prob_thresh=0.5,
        clear_nBorder=True,
        min_nSize=4096,
        tophat_size=3,
        tophat_sigma=1,
        tophat_tresh_coeff=1.25,
        min_cSize=32,
        ):
    
    def _get_tophat(plane):
        tophat = white_tophat(plane, footprint=disk(tophat_size)) 
        tophat = gaussian(tophat, sigma=tophat_sigma, preserve_range=True)
        return tophat
    
    # Load & format data
    stack, metadata = open_stack(path)
    rscale, rslice = format_stack(stack, metadata)
    ratio = metadata["vY"] / metadata["vZ"]
    
    # Predict
    probs = predict(rscale, rslice)
    probs = resize(probs, stack.shape, order=1)
    probs = gaussian(probs, sigma=(
        prob_sigma * ratio, prob_sigma, prob_sigma
        ))
    
    # Nuclei
    nMask = probs > prob_thresh
    if clear_nBorder:
        nMask = clear_border(nMask)
    nMask = remove_small_objects(nMask, min_size=min_nSize)
    nLabels = label(nMask)
    nOutlines = binary_dilation(nMask) ^ nMask

    # Tophat transform
    tophat = Parallel(n_jobs=-1)(
        delayed(_get_tophat)(plane)
        for plane in stack
        )
    tophat = np.stack(tophat)
    
    # Chromocenters
    cMask = np.zeros_like(tophat, dtype=bool)
    for lab in np.unique(nLabels):
        if lab > 0:
            idx = (nLabels == lab)
            values = tophat[idx]
            thresh = threshold_otsu(values)
            cMask[idx] = (values > thresh * tophat_tresh_coeff)
    cMask = remove_small_objects(cMask, min_size=min_cSize)
    cLabels = label(cMask)
    cOutlines = binary_dilation(cMask) ^ cMask
    
    # Save
    save_path = Path(data_path, path.stem)
    save_path.mkdir(exist_ok=True)
    
    io.imsave(
        save_path / (path.name + "_nMask.tif"), 
        nMask.astype("uint8") * 255, check_contrast=False
        )
    io.imsave(
        save_path / (path.name + "_cMask.tif"), 
        cMask.astype("uint8") * 255, check_contrast=False
        )
    
def test(path):
    
    global stack, metadata, probs
    
    # Load & format data
    stack, metadata = open_stack(path)
    rscale, rslice = format_stack(stack, metadata)
    ratio = metadata["vY"] / metadata["vZ"]
    
    # Predict
    probs = predict(rscale, rslice)
    
    
    # probs = resize(probs, stack.shape, order=1)
    # probs = gaussian(probs, sigma=(
    #     prob_sigma * ratio, prob_sigma, prob_sigma
    #     ))
        
#%% Execute -------------------------------------------------------------------

if __name__ == "__main__":
    for path in data_path.iterdir():
        if path.suffix == ".nd2":
            if path.stem == "KASind1": # dev
            # if path.stem == "KZLind1": # dev
            # if path.stem == "KASind1003": # dev
            # if path.stem == "KZLind1002": # dev
            # if path.stem == "KZLind3001": # dev
                test(path)
                
#%%

from skimage.morphology import ball
from skimage.feature import peak_local_max
from skimage.segmentation import expand_labels

print("get_nMask :", end='')
t0 = time.time()
  
lmax = peak_local_max(
    probs, min_distance=5, threshold_abs=0.15, exclude_border=False)

sMask = probs > 0.4 # parameter(s)
sMask = remove_small_objects(sMask, min_size=8) # parameter(s)

sMax = np.zeros_like(probs, dtype=int)
sMax[(lmax[:, 0], lmax[:, 1], lmax[:, 2])] = 1

sLabels = label(sMax)
sLabels = expand_labels(sLabels, distance=10) 
sLabels[sMask == 0] = 0
sLabels = expand_labels(sLabels, distance=10)

sMax = resize(sMax, stack.shape, order=0)
sLabels = resize(sLabels, stack.shape, order=0)

nMask = resize(probs, stack.shape, order=1) > 0.25 # parameter(s)
nMask = remove_small_objects(nMask, min_size=4096) # parameter(s)
sLabels[nMask == 0] = 0

t1 = time.time()
print(f" {(t1-t0):<5.2f}s")

# Display 
scale = [metadata["vZ"] / metadata["vY"], 1, 1]
viewer = napari.Viewer()
viewer.add_image(stack, scale=scale)
viewer.add_image(
    nMask, scale=scale, 
    rendering='attenuated_MIP', colormap='magenta', opacity=0.75
    )
viewer.add_labels(sLabels, scale=scale, opacity=0.75)
viewer.add_image(sMax, scale=scale, opacity=0.75)