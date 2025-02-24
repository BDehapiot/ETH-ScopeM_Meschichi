#%% Imports -------------------------------------------------------------------

import nd2
import napari
import numpy as np
import pandas as pd
from skimage import io
from pathlib import Path
import segmentation_models as sm
from joblib import Parallel, delayed

# bdtools
from bdtools.norm import norm_gcn, norm_pct

# # bdmodel
# from bdmodel.predict import predict

# Skimage
from skimage.feature import peak_local_max
from skimage.transform import rescale, resize
from skimage.measure import label, regionprops
from skimage.filters import gaussian, threshold_otsu
from skimage.segmentation import clear_border, expand_labels
from skimage.morphology import (
    disk, remove_small_objects, white_tophat, binary_dilation
    )

# Scipy
from scipy.ndimage import distance_transform_edt

#%% Functions (GPU) -----------------------------------------------------------

def limit_vram(vram):

    import tensorflow as tf    
    gpus = tf.config.experimental.list_physical_devices('GPU')
    memory_config = tf.config.experimental\
        .VirtualDeviceConfiguration(memory_limit=vram)
    
    if gpus:
        try:
            tf.config.experimental\
                .set_virtual_device_configuration(gpus[0], [memory_config])
        except RuntimeError as e:
            print(e)

#%% Functions (open_stack) ----------------------------------------------------

def open_stack(path, metadata=True):
    
    # Read nd2 file
    with nd2.ND2File(path) as ndfile:
        stack = ndfile.asarray()
        nZ, nY, nX = stack.shape
        vY, vX, vZ = ndfile.voxel_size()
    
    if metadata:
        
        metadata = {
            "nZ" : nZ, "nY" : nY, "nX" : nX, 
            "vZ" : vZ, "vY" : vY, "vX" : vX,
            }
    
        return stack, metadata
    
    return stack

#%% Functions (format_stack) --------------------------------------------------

def format_stack(stack, metadata):
           
    # Rescale & reslice (isotropic voxel)
    ratio = metadata["vY"] / metadata["vZ"]
    rscale = rescale(stack, (1, ratio, ratio), order=0)
    rslice = np.swapaxes(rscale, 0, 1)
        
    return rscale, rslice

#%% Functions (segment) -------------------------------------------------------

def segment(
        path,
        # nMask
        lmax_dist=5,
        lmax_prom=0.15,
        prob_thresh=0.25,
        clear_nBorder=True,
        min_nSize=4096,
        # cMask
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
    
    # Predict
    rscale_prds = predict(
        rscale, Path("bdmodel", "model_rscale_128"), patch_overlap=0)
    rslice_prds = predict(
        rslice, Path("bdmodel", "model_rslice_128"), patch_overlap=0)
    
    # Merge predictions
    rslice_prds = np.swapaxes(rslice_prds, 0, 1)
    prds = (rscale_prds + rslice_prds) / 2
    
    # Nuclei
    lmax = peak_local_max(
        prds, exclude_border=False,
        min_distance=lmax_dist, threshold_abs=lmax_prom,
        )
    pMask = prds > 0.4 # parameter(s)
    pMask = remove_small_objects(pMask, min_size=8) # parameter(s)
    pMax = np.zeros_like(prds, dtype=int)
    pMax[(lmax[:, 0], lmax[:, 1], lmax[:, 2])] = 1
    nLabels = label(pMax)
    nLabels = expand_labels(nLabels, distance=10) 
    nLabels[pMask == 0] = 0
    nLabels = expand_labels(nLabels, distance=10)
    pMax = resize(pMax, stack.shape, order=0)
    nLabels = resize(nLabels, stack.shape, order=0)
    nMask = resize(prds, stack.shape, order=1) > prob_thresh
    if clear_nBorder:
        nMask = clear_border(nMask)
    nMask = remove_small_objects(nMask, min_size=min_nSize)
    nLabels[nMask == 0] = 0
    
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
    
    # Format outputs
    outputs = {
        "stack"    : stack,
        "metadata" : metadata,
        "prds"     : prds.astype("float32"),
        "nLabels"  : nLabels.astype("uint16"),
        "cLabels"  : cLabels.astype("uint16"),
        "tophat"   : tophat.astype("float32"),
        }
    
    return outputs

#%% Functions (Measure) -------------------------------------------------------

def measure(outputs, rFactor=0.5):

    # Get variables
    stack = outputs["stack"]
    nLabels = outputs["nLabels"]
    cLabels = outputs["cLabels"]
    metadata = outputs["metadata"]
    
    # Isotropic stacks (1st rescaling)
    ratio = metadata["vZ"] / metadata["vY"]
    stack = rescale(stack, (ratio * rFactor, rFactor, rFactor), order=0)
    nLabels = rescale(nLabels, (ratio * rFactor, rFactor, rFactor), order=0)
    cLabels = rescale(cLabels, (ratio * rFactor, rFactor, rFactor), order=0)
    voxSize = metadata["vY"] / rFactor
    
    # nData -------------------------------------------------------------------

    nData = {
        
        # Nuclei
        "nLabel"   : [],
        "nVolume"  : [],
        "nCtrd"    : [],
        "nMajor"   : [],
        "nMinor"   : [],
        "nMMRatio" : [],
        
        # Chromocenters
        "n_cLabels"  : [],
        "n_cNumber"  : [],
        "n_cVolume"  : [],
        "n_cnRatio"  : [],
        
        }

    for nProps in regionprops(nLabels):
        
        # Nuclei
        nData["nLabel"].append(nProps.label)
        nVolume = nProps.area * (voxSize ** 3)
        nData["nVolume"].append(nVolume)
        nCtrd = nProps.centroid
        nCtrd = [int(ctrd) for ctrd in nCtrd]
        nData["nCtrd"].append(nCtrd)
        try:
            nMajor = nProps.axis_major_length * voxSize
            nMinor = nProps.axis_minor_length * voxSize
            nMMRatio = nMinor / nMajor
            nData["nMajor"].append(nMajor)
            nData["nMinor"].append(nMinor)
            nData["nMMRatio"].append(nMMRatio)
        except ValueError:
            nData["nMajor"].append(np.nan)
            nData["nMinor"].append(np.nan)
            nData["nMMRatio"].append(np.nan)

        # Chromocenters
        unique = np.unique(cLabels[nLabels == nProps.label])
        unique = unique[unique != 0]
        nData["n_cLabels"].append(unique)
        nData["n_cNumber"].append(unique.shape[0])
        cMask = cLabels > 0
        cVolume = np.sum(cMask[nLabels == nProps.label]) * (voxSize ** 3)
        nData["n_cVolume"].append(cVolume)
        nData["n_cnRatio"].append(cVolume / nVolume)

    # EDMs --------------------------------------------------------------------

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
    EDMb = resize(EDMb, nLabels.shape, order=0)
    EDMc = resize(EDMc, nLabels.shape, order=0)
    
    # cData -------------------------------------------------------------------

    cData = {
        
        # Chromocenters
        "cLabel"   : [],
        "cVolume"  : [],
        "cCtrd"    : [],
        "cMajor"   : [],
        "cMinor"   : [],
        "cMMRatio" : [],
        "cInt"     : [],
        "cEDMb"    : [],
        "cEDMc"    : [], 
        
        # Nuclei
        "c_nLabel"   : [],
        
        }

    for cProps in regionprops(cLabels, intensity_image=stack):
        
        # Chromocenters
        cData["cLabel"].append(cProps.label)
        cVolume = cProps.area * (voxSize ** 3)
        cData["cVolume"].append(cVolume)
        cCtrd = cProps.centroid
        cCtrd = [int(ctrd) for ctrd in cCtrd]
        cData["cCtrd"].append(cCtrd)
        try:
            cMajor = cProps.axis_major_length * voxSize
            cMinor = cProps.axis_minor_length * voxSize
            cMMRatio = cMinor / cMajor
            cData["cMajor"].append(cMajor)
            cData["cMinor"].append(cMinor)
            cData["cMMRatio"].append(cMMRatio)
        except ValueError:
            cData["cMajor"].append(np.nan)
            cData["cMinor"].append(np.nan)
            cData["cMMRatio"].append(np.nan)
        cData["cInt"].append(cProps.intensity_mean) # To be discussed
        cEDMb = EDMb[cCtrd[0], cCtrd[1], cCtrd[2]] * voxSize
        cEDMc = EDMc[cCtrd[0], cCtrd[1], cCtrd[2]] * voxSize
        cData["cEDMb"].append(cEDMb)
        cData["cEDMc"].append(cEDMc)
        
        # Associated nuclei
        cData["c_nLabel"].append(np.max(nLabels[cLabels == cProps.label]))

    # Outputs -----------------------------------------------------------------

    # Correct centroids
    for i, nCtrd in enumerate(nData["nCtrd"]):
        nData["nCtrd"][i] = [int(c / rFactor) for c in nCtrd]
    for i, cCtrd in enumerate(cData["cCtrd"]):
        cData["cCtrd"][i] = [int(c / rFactor) for c in cCtrd]

    # Append outputs
    outputs["nData"] = pd.DataFrame(nData)
    outputs["cData"] = pd.DataFrame(cData)

    return outputs

#%% Functions (save) ----------------------------------------------------------

def save(path, outputs):  
    save_path = path.parent / (path.stem)
    save_path.mkdir(parents=True, exist_ok=True)
    
    # Dataframes
    outputs["nData"].to_csv(
        save_path / f"{path.stem}_nData.csv", index=False, float_format='%.3f')
    outputs["cData"].to_csv(
        save_path / f"{path.stem}_cData.csv", index=False, float_format='%.3f')
    
    # Images
    io.imsave(
        save_path / f"{path.stem}_stack.tif",
        outputs["stack"], check_contrast=False,
        )
    io.imsave(
        save_path / f"{path.stem}_prds.tif",
        outputs["prds"], check_contrast=False,
        )
    io.imsave(
        save_path / f"{path.stem}_nLabels.tif",
        outputs["nLabels"], check_contrast=False,
        )
    io.imsave(
        save_path / f"{path.stem}_cLabels.tif",
        outputs["cLabels"], check_contrast=False,
        )
    io.imsave(
        save_path / f"{path.stem}_tophat.tif",
        outputs["tophat"], check_contrast=False,
        )

#%% Functions (display) -------------------------------------------------------

def display(stack, metadata, nLabels, cLabels, tophat):

    def get_outlines(labels):
        mask = labels > 0
        outlines = np.zeros_like(mask)
        for z, img in enumerate(mask):
            outlines[z, ...] = binary_dilation(img) ^ img  
        return outlines
    
    # Scale
    scale = [metadata["vZ"] / metadata["vY"], 1, 1]
    
    # Outlines
    nOutlines = get_outlines(nLabels)
    cOutlines = get_outlines(cLabels)
    
    # Viewer (labels)
    viewer0 = napari.Viewer()
    viewer0.add_image(stack, scale=scale)
    viewer0.add_labels(nLabels, scale=scale,
        blending="additive")
    viewer0.add_labels(cLabels, scale=scale, 
        blending="translucent")
    
    # Viewer (segmentation)
    viewer1 = napari.Viewer()
    viewer1.add_image(stack, scale=scale,
      blending="additive", opacity=1.00, gamma=0.75, colormap="inferno")
    viewer1.add_image(tophat, scale=scale,
      blending="additive", opacity=1.00, gamma=1.50, colormap="inferno")
    viewer1.add_image(nOutlines, scale=scale,
      blending="additive", opacity=0.25, gamma=1.00, colormap="gray")
    viewer1.add_image(cOutlines, scale=scale,
      blending="additive", opacity=0.25, gamma=1.00, colormap="gray")
    