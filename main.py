#%% Imports -------------------------------------------------------------------

import time
import napari
import numpy as np
from pathlib import Path
from joblib import Parallel, delayed
from functions import open_stack, format_stack, predict

#%% Inputs --------------------------------------------------------------------

# Paths
# local_path = Path('D:/local_Meschichi/data')
local_path = Path(Path.cwd(), 'data', 'local') 
stack_paths = []
for path in local_path.iterdir():
    if path.suffix == ".nd2":
        stack_paths.append(path)

# Select stack
idx = 7

#%% Process -------------------------------------------------------------------

# Skimage
from skimage.measure import label
from skimage.transform import resize
from skimage.segmentation import clear_border
from skimage.filters import gaussian, threshold_otsu
from skimage.morphology import (
    disk, remove_small_objects, binary_dilation, white_tophat
    )

# -----------------------------------------------------------------------------

# GPU
# vram = 2048 # Limit vram (None to deactivate)
vram = None # Limit vram (None to deactivate)

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

#­ -----------------------------------------------------------------------------

if vram:
    from functions import limit_vram
    limit_vram(vram)

# -----------------------------------------------------------------------------

def get_nMask(stack, metadata, prob_sigma, prob_thresh):
    
    # Format data
    rscale, rslice = format_stack(stack, metadata)
    ratio = metadata["vY"] / metadata["vZ"]
    
    # Predict
    probs = predict(rscale, rslice)
    probs = resize(probs, stack.shape, order=1)
    probs = gaussian(probs, sigma=(
        prob_sigma * ratio, prob_sigma, prob_sigma
        ))
    
    # Nuclei mask (nMask)
    nMask = probs > prob_thresh
    if clear_nBorder:
        nMask = clear_border(nMask)
    nMask = remove_small_objects(nMask, min_size=min_nSize)
    
    # Nuclei labels (nLabels)
    nLabels = label(nMask)
    
    # Nuclei outlines (nOutlines)
    nOutlines = binary_dilation(nMask) ^ nMask

    return nMask, nLabels, nOutlines

# -----------------------------------------------------------------------------

def get_cMask(stack, nLabels, tophat_size, tophat_sigma):
    
    def _get_tophat(plane):
        tophat = white_tophat(plane, footprint=disk(tophat_size)) 
        tophat = gaussian(tophat, sigma=tophat_sigma, preserve_range=True)
        return tophat
       
    # Tophat transform
    tophat = Parallel(n_jobs=-1)(
        delayed(_get_tophat)(plane)
        for plane in stack
        )
    tophat = np.stack(tophat)
    
    # Chromocenter mask (cMask)
    cMask = np.zeros_like(tophat, dtype=bool)
    for lab in np.unique(nLabels):
        if lab > 0:
            idx = (nLabels == lab)
            values = tophat[idx]
            thresh = threshold_otsu(values)
            cMask[idx] = (values > thresh * tophat_tresh_coeff)
    cMask = remove_small_objects(cMask, min_size=min_cSize)
    
    # Chromocenter outlines (cOutlines)
    cOutlines = binary_dilation(cMask) ^ cMask
    
    return cMask, cOutlines, tophat   

# -----------------------------------------------------------------------------

stack, metadata = open_stack(stack_paths[idx])

print("get_nMask :", end='')
t0 = time.time()

nMask, nLabels, nOutlines = get_nMask(stack, metadata, prob_sigma, prob_thresh)

t1 = time.time()
print(f" {(t1-t0):<5.2f}s")

# -----------------------------------------------------------------------------

print("get_cMask :", end='')
t0 = time.time()

cMask, cOutlines, tophat  = get_cMask(stack, nLabels, tophat_size, tophat_sigma)

t1 = time.time()
print(f" {(t1-t0):<5.2f}s")

#%% Display -------------------------------------------------------------------

scale = [metadata["vZ"] / metadata["vY"], 1, 1]

# Check segmentation 2D
viewer = napari.Viewer()
viewer.add_image(stack, scale=scale, colormap='plasma')
viewer.add_image(tophat, scale=scale, colormap='plasma')
viewer.add_image(nOutlines, scale=scale, blending='additive', opacity=0.25)
viewer.add_image(cOutlines, scale=scale, blending='additive', opacity=0.5)

# Check segmentation 3D
viewer = napari.Viewer()
viewer.add_image(stack, scale=scale)
viewer.add_image(tophat, scale=scale)
viewer.add_image(nMask, scale=scale, rendering='attenuated_MIP', colormap='magenta', opacity=0.75)
viewer.add_image(cMask, scale=scale, rendering='attenuated_MIP', colormap='green', opacity=0.5)
viewer.dims.ndisplay = 3
